"""Offside detector for the schematic VAR images.

The grass pattern and the two team colours change from image to image.  The
analysis below is kept in NumPy so that the submission does not depend on a
particular OpenCV version.  OpenCV images (BGR) and RGB arrays work
identically because the colours are read from the legend in the image itself.
"""

import numpy as np


PLAYER_RADIUS = 12
METRES_PER_PIXEL = 0.127
REFERENCE_WIDTH = 900
REFERENCE_HEIGHT = 600


def _normalise_image(image):
    """Map scaled challenge images to the coordinate system used for scoring.

    The generated pitch uses a fixed physical scale.  Normalising first keeps
    player size, geometric thresholds, and metres-per-pixel consistent even
    when the image supplied to the function has been resized.
    """

    height, width = image.shape[:2]
    if (height, width) == (REFERENCE_HEIGHT, REFERENCE_WIDTH):
        return image

    # Nearest-neighbour sampling preserves the flat colours used by the
    # schematic images and needs no OpenCV dependency.
    ys = np.rint(
        (np.arange(REFERENCE_HEIGHT) + 0.5) * height / REFERENCE_HEIGHT - 0.5
    ).astype(np.intp)
    xs = np.rint(
        (np.arange(REFERENCE_WIDTH) + 0.5) * width / REFERENCE_WIDTH - 0.5
    ).astype(np.intp)
    np.clip(ys, 0, height - 1, out=ys)
    np.clip(xs, 0, width - 1, out=xs)
    return image[ys[:, None], xs[None, :]]


def _component_boxes(mask, min_area=1, x0=0, x1=None, y0=0, y1=None):
    """Return (area, left, top, right, bottom) for connected components."""

    height, width = mask.shape
    x1 = width if x1 is None else min(width, x1)
    y1 = height if y1 is None else min(height, y1)
    x0 = max(0, x0)
    y0 = max(0, y0)

    seen = np.zeros_like(mask, dtype=bool)
    components = []

    for y in range(y0, y1):
        for x in range(x0, x1):
            if not mask[y, x] or seen[y, x]:
                continue

            stack = [(x, y)]
            seen[y, x] = True
            area = 0
            left = right = x
            top = bottom = y

            while stack:
                px, py = stack.pop()
                area += 1
                left = min(left, px)
                right = max(right, px)
                top = min(top, py)
                bottom = max(bottom, py)

                for nx, ny in (
                    (px - 1, py),
                    (px + 1, py),
                    (px, py - 1),
                    (px, py + 1),
                    (px - 1, py - 1),
                    (px + 1, py - 1),
                    (px - 1, py + 1),
                    (px + 1, py + 1),
                ):
                    if (
                        x0 <= nx < x1
                        and y0 <= ny < y1
                        and mask[ny, nx]
                        and not seen[ny, nx]
                    ):
                        seen[ny, nx] = True
                        stack.append((nx, ny))

            if area >= min_area:
                components.append((area, left, top, right, bottom))

    return components


def _find_legend(image):
    """Find the black legend rectangle and return its bounding box."""

    dark = np.max(image, axis=2) < 70
    boxes = _component_boxes(dark, min_area=300)
    if not boxes:
        return None

    boxes.sort(key=lambda box: box[0], reverse=True)
    for area, left, top, right, bottom in boxes:
        if right - left >= 80 and bottom - top >= 30:
            return left, top, right, bottom
    _, left, top, right, bottom = boxes[0]
    return left, top, right, bottom


def _dominant_colour(block):
    """Return the most frequent pixel in a small legend swatch sample."""

    pixels = block.reshape(-1, 3)
    colours, counts = np.unique(pixels, axis=0, return_counts=True)
    return tuple(int(value) for value in colours[np.argmax(counts)])


def _legend_colours(image, legend):
    """Read attacker and defender swatch colours from the legend."""

    if legend is None:
        return [], []

    left, top, right, bottom = legend
    box_width = right - left + 1
    box_height = bottom - top + 1

    # Locate swatches proportionally within the legend instead of relying on
    # one exact border thickness or pixel position.
    x_centres = (
        left + int(round(box_width * 0.115)),
        left + int(round(box_width * 0.191)),
    )
    patch_x = max(1, int(round(box_width * 0.025)))
    patch_y = max(1, int(round(box_height * 0.09)))

    def read_row(y_fraction):
        centre_y = top + int(round(box_height * y_fraction))
        y_start = max(0, centre_y - patch_y)
        y_stop = min(image.shape[0], centre_y + patch_y + 1)
        samples = []
        for centre_x in x_centres:
            x_start = max(0, centre_x - patch_x)
            x_stop = min(image.shape[1], centre_x + patch_x + 1)
            if y_stop <= y_start or x_stop <= x_start:
                continue
            samples.append(
                _dominant_colour(image[y_start:y_stop, x_start:x_stop])
            )

        if not samples:
            return []
        result = []
        for colour in samples:
            if not any(
                np.max(np.abs(np.asarray(colour) - np.asarray(old))) <= 20
                for old in result
            ):
                result.append(colour)
        return result

    return read_row(0.303), read_row(0.682)


def _is_white(colour):
    colour = np.asarray(colour)
    return int(colour.min()) >= 220 and int(colour.max() - colour.min()) <= 35


def _unique_colours(first, second):
    """Colours occurring in one team swatch but not the other."""

    def only_these(source, other):
        result = []
        for colour in source:
            if not any(
                np.max(np.abs(np.asarray(colour) - np.asarray(other_colour))) <= 40
                for other_colour in other
            ):
                result.append(colour)
        return result

    return only_these(first, second), only_these(second, first)


def _build_colour_masks(image, colours, tolerance=48):
    """Assign each pixel to its nearest legend colour, without overlap."""

    pixels = image.astype(np.int16)
    distances = np.stack(
        [
            np.max(np.abs(pixels - np.asarray(colour, dtype=np.int16)), axis=2)
            for colour in colours
        ],
        axis=0,
    )
    nearest = np.argmin(distances, axis=0)
    minimum = np.min(distances, axis=0)
    return {
        colour: (nearest == index) & (minimum <= tolerance)
        for index, colour in enumerate(colours)
    }


def _goal_is_at_top(image, legend):
    """Infer the attacking direction from the main white pitch component."""

    channel_min = np.min(image, axis=2)
    channel_range = np.max(image, axis=2) - channel_min
    white = (channel_min >= 220) & (channel_range <= 35)
    if legend is not None:
        left, top, right, bottom = legend
        white[max(0, top - 2):bottom + 3, max(0, left - 2):right + 3] = False

    components = _component_boxes(white, min_area=150)
    if not components:
        return True

    # Pitch markings make the largest connected white object.  The goal and
    # penalty-area drawing occupy the half containing the attacked goal.
    _, _, top, _, bottom = max(components, key=lambda component: component[0])
    return (top + bottom) / 2 < image.shape[0] / 2


def _window_sums(mask, size):
    """Sum a binary mask over every valid square window."""

    integral = np.pad(mask.astype(np.int32), ((1, 0), (1, 0)))
    integral = integral.cumsum(axis=0).cumsum(axis=1)
    return (
        integral[size:, size:]
        - integral[:-size, size:]
        - integral[size:, :-size]
        + integral[:-size, :-size]
    )


def _circle_centres(mask, threshold=330):
    """Find player-sized circular blobs, suppressing duplicate maxima."""

    size = 2 * PLAYER_RADIUS + 1
    sums = _window_sums(mask, size)

    candidate_y, candidate_x = np.nonzero(sums >= threshold)
    if candidate_y.size == 0:
        return []
    values = sums[candidate_y, candidate_x]
    order = np.argsort(values)[::-1]

    yy, xx = np.ogrid[-PLAYER_RADIUS:PLAYER_RADIUS + 1,
                      -PLAYER_RADIUS:PLAYER_RADIUS + 1]
    disc = xx * xx + yy * yy <= PLAYER_RADIUS * PLAYER_RADIUS
    required_disc_fill = int(disc.sum() * 0.68)

    selected = []
    for index in order:
        x = int(candidate_x[index]) + PLAYER_RADIUS
        y = int(candidate_y[index]) + PLAYER_RADIUS
        value = int(values[index])
        if all(
            (x - old_x) ** 2 + (y - old_y) ** 2 >= 15 ** 2
            for _, old_x, old_y in selected
        ):
            patch = mask[
                y - PLAYER_RADIUS:y + PLAYER_RADIUS + 1,
                x - PLAYER_RADIUS:x + PLAYER_RADIUS + 1,
            ]
            if patch.shape == disc.shape and int(patch[disc].sum()) >= required_disc_fill:
                selected.append((value, x, y))
    return selected


def _colour_counts(masks, x, y):
    y0 = max(0, y - PLAYER_RADIUS)
    y1 = min(masks[0].shape[0], y + PLAYER_RADIUS + 1)
    x0 = max(0, x - PLAYER_RADIUS)
    x1 = min(masks[0].shape[1], x + PLAYER_RADIUS + 1)
    return [int(mask[y0:y1, x0:x1].sum()) for mask in masks]


def _classify_white_blob(x, y, palettes, unique_masks, colour_masks):
    """Classify a white circle using the non-shared half of a uniform."""

    scores = []
    for masks in unique_masks:
        scores.append(sum(_colour_counts(masks, x, y)) if masks else 0)

    if scores[0] > scores[1] + 20:
        return 0
    if scores[1] > scores[0] + 20:
        return 1

    # A solid white team has a one-colour swatch, while the opposing
    # split-colour team has an additional colour that is absent here.
    lengths = [len(palettes[0]), len(palettes[1])]
    if lengths[0] < lengths[1]:
        return 0
    if lengths[1] < lengths[0]:
        return 1

    # Last-resort tie breaker: compare the amount of each complete palette.
    totals = []
    for palette in palettes:
        totals.append(
            sum(_colour_counts([colour_masks[colour] for colour in palette], x, y))
        )
    return int(totals[1] > totals[0])


def _white_edges_for_team(
    white_mask,
    team_index,
    palettes,
    unique_masks,
    colour_masks,
    goal_at_top,
):
    """Get top edges of white players, excluding white pitch markings."""

    edges = []
    for _, x, y in _circle_centres(white_mask, threshold=330):
        owner = _classify_white_blob(
            x, y, palettes, unique_masks, colour_masks
        )
        if owner == team_index:
            edges.append(y - PLAYER_RADIUS if goal_at_top else y + PLAYER_RADIUS)
    return edges


def _coloured_edges(colour_masks, goal_at_top, min_area=55):
    """Get circle top edges from a non-white team colour."""

    edges = []
    for mask in colour_masks:
        for area, left, top, right, bottom in _component_boxes(
            mask, min_area=min_area
        ):
            component_height = bottom - top + 1
            component_width = right - left + 1
            box_area = component_height * component_width
            fill_ratio = area / box_area

            # Full player discs and vertical half-discs are both about 25 px
            # tall.  Bounds plus fill ratio reject streaks and colour noise.
            player_shaped = (
                12 <= component_height <= 34
                and 7 <= component_width <= 34
                and area <= 720
                and 0.45 <= fill_ratio <= 0.88
            )
            if player_shaped:
                edges.append(top if goal_at_top else bottom)
    return edges


def _fallback_edges(
    mask,
    team_index,
    palettes,
    unique_masks,
    colour_masks,
    goal_at_top,
):
    """Fallback for an unusual palette or a partially occluded player."""

    edges = []
    for _, x, y in _circle_centres(mask, threshold=280):
        if _classify_white_blob(x, y, palettes, unique_masks, colour_masks) == team_index:
            edges.append(y - PLAYER_RADIUS if goal_at_top else y + PLAYER_RADIUS)
    return edges


def offside_detection(img):
    """Return ``Onside!`` or ``Offside!\n<distance in metres>``."""

    image = np.asarray(img)
    if image.ndim != 3 or image.shape[2] < 3:
        raise ValueError("img must be a colour image")
    if image.shape[0] == 0 or image.shape[1] == 0:
        raise ValueError("img must not be empty")
    image = _normalise_image(image[..., :3])

    legend = _find_legend(image)
    palettes = list(_legend_colours(image, legend))
    if len(palettes) != 2 or not palettes[0] or not palettes[1]:
        raise ValueError("could not locate the attacker/defender legend")
    goal_at_top = _goal_is_at_top(image, legend)

    # De-duplicate swatch entries while retaining the original team palettes.
    all_colours = []
    for palette in palettes:
        for colour in palette:
            if not any(
                np.max(np.abs(np.asarray(colour) - np.asarray(old))) <= 20
                for old in all_colours
            ):
                all_colours.append(colour)
    colour_masks = _build_colour_masks(image, all_colours)

    # Remove the legend from every mask.  Its swatches are otherwise large
    # enough to look like players to the square-window detector.
    if legend is not None:
        left, top, right, bottom = legend
        for mask in colour_masks.values():
            mask[max(0, top - 1):bottom + 2, max(0, left - 1):right + 2] = False

    unique_colours = _unique_colours(palettes[0], palettes[1])
    unique_masks = [
        [colour_masks[colour] for colour in unique_colours[0]],
        [colour_masks[colour] for colour in unique_colours[1]],
    ]

    # Non-white colours never occur in pitch markings, so their component
    # bounds give a particularly reliable player top edge.
    coloured_edges = []
    for team_index in range(2):
        nonwhite = [
            mask
            for colour, mask in zip(unique_colours[team_index], unique_masks[team_index])
            if not _is_white(colour)
        ]
        coloured_edges.append(_coloured_edges(nonwhite, goal_at_top))

    # White discs need a separate detector because they touch the white pitch
    # lines.  If a team has a non-white swatch half, use that half instead.
    white_colours = [colour for colour in all_colours if _is_white(colour)]
    white_mask = np.zeros(image.shape[:2], dtype=bool)
    for colour in white_colours:
        white_mask |= colour_masks[colour]
    white_edges = [[], []]
    if white_colours and white_mask.any():
        for team_index in range(2):
            if not coloured_edges[team_index]:
                white_edges[team_index] = _white_edges_for_team(
                    white_mask,
                    team_index,
                    palettes,
                    unique_masks,
                    colour_masks,
                    goal_at_top,
                )

    team_edges = [
        coloured_edges[0] or white_edges[0],
        coloured_edges[1] or white_edges[1],
    ]

    # The examples show the attacking direction toward the top goal.  A
    # fallback is useful if a noisy image made one of the colour components
    # too small to pass the component filter.
    all_player_mask = np.zeros(image.shape[:2], dtype=bool)
    for mask in colour_masks.values():
        all_player_mask |= mask
    for team_index in range(2):
        if not team_edges[team_index]:
            team_edges[team_index] = _fallback_edges(
                all_player_mask,
                team_index,
                palettes,
                unique_masks,
                colour_masks,
                goal_at_top,
            )

    if not team_edges[0]:
        return "Onside!"
    if not team_edges[1]:
        raise ValueError("could not detect any defenders")

    if goal_at_top:
        defender_line = min(team_edges[1])
        offside_edges = [edge for edge in team_edges[0] if edge < defender_line]
    else:
        defender_line = max(team_edges[1])
        offside_edges = [edge for edge in team_edges[0] if edge > defender_line]
    if not offside_edges:
        return "Onside!"

    # Select the forward-most point that is beyond, but nearest to, the line.
    closest_attacker_edge = (
        max(offside_edges) if goal_at_top else min(offside_edges)
    )
    distance = abs(defender_line - closest_attacker_edge) * METRES_PER_PIXEL
    return f"Offside!\n{distance:.2f}"
    
