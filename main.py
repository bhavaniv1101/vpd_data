import math
import random
from dataclasses import dataclass
import cv2
from google.colab.patches import cv2_imshow
import skimage.exposure
import numpy as np
from numpy.random import default_rng
from PIL import Image, ImageDraw

# NOTE: Origin is at top left corner, +ve x-axis points **right**, +ve y-axis points **down**.
# This means angles increase in **clockwise** direction from +ve x-axis.


SIZE = 256


@dataclass(frozen=True)
class Point:
    """Point in the x-y plane, specified through its Cartesian coordinates."""

    x: float
    y: float

    def isclose(self, other: "Point") -> bool:
        x_isclose = math.isclose(self.x, other.x, abs_tol=1e-12)
        y_isclose = math.isclose(self.y, other.y, abs_tol=1e-12)
        return x_isclose and y_isclose

    def translated(self, distance: float, angle_rad: float) -> "Point":
        """Return result of translation by given distance at given angle."""
        # NOTE: Since +ve x-axis points right and +ve y-axis points down, the latter is 90 degrees
        # clockwise from the former, unlike in the "normal" coordinate system. So `angle_rad` must
        # increase in clockwise direction from a value of `0.0` along the +ve x-axis.
        # NOTE: Negative distance is allowed: translating by `distance` along `angle_rad` is
        # equivalent to translating by `-distance` along `math.pi + angle_rad`.
        new_point_x = self.x + distance * math.cos(angle_rad)
        new_point_y = self.y + distance * math.sin(angle_rad)
        return Point(new_point_x, new_point_y)

    def rotated(self, angle_rad: float, center: "Point") -> "Point":
        """Return result of rotation by given angle around given center."""
        diff = Point(self.x - center.x, self.y - center.y)
        cos = math.cos(angle_rad)
        sin = math.sin(angle_rad)
        new_point_x = center.x + diff.x * cos - diff.y * sin
        new_point_y = center.y + diff.x * sin + diff.y * cos
        return Point(new_point_x, new_point_y)


@dataclass(frozen=True)
class LineSegment:
    """Line segment in the x-y plane, specified through its start and end points."""

    start: Point
    end: Point

    def length(self) -> float:
        """Return length."""
        return math.sqrt(
            (self.end.x - self.start.x) ** 2 + (self.end.y - self.start.y) ** 2
        )

    def angle(self) -> float:
        """Return angle (in radians) with respect to +ve x-axis."""
        return math.atan2(self.end.y - self.start.y, self.end.x - self.start.x)

    def midpoint(self) -> Point:
        """Return midpoint."""
        return Point((self.start.x + self.end.x) / 2, (self.start.y + self.end.y) / 2)

    def point_on_perpendicular_bisector(self, distance_from_midpoint) -> Point:
        """Return point on perpendicular bisector at given distance from midpoint."""
        return self.midpoint().translated(
            distance_from_midpoint,
            self.angle() + math.pi / 2,
        )

    def parallel(
        self, midpoint_of_parallel: Point, length_of_parallel: float
    ) -> "LineSegment":
        """Return parallel line segment with given midpoint and length."""
        half_length_of_parallel = length_of_parallel / 2
        angle_rad = self.angle()
        start = midpoint_of_parallel.translated(
            distance=-half_length_of_parallel, angle_rad=angle_rad
        )
        end = midpoint_of_parallel.translated(
            distance=half_length_of_parallel, angle_rad=angle_rad
        )
        return LineSegment(start, end)

    def rotated(self, angle_rad: float, center: Point) -> "LineSegment":
        """Return result of rotation by given angle around given center."""
        new_start = self.start.rotated(angle_rad, center)
        new_end = self.end.rotated(angle_rad, center)
        return LineSegment(new_start, new_end)


def circle_center() -> Point:
    """Return center of circle inscribed in square with vertices `(0, 0)`, `(S, 0)`, `(S, S)`,
    and `(0, S)`. Here `S = SIZE`.
    """
    return Point(SIZE / 2, SIZE / 2)


def circle_radius() -> float:
    """Return radius of circle inscribed in square with vertices `(0, 0)`, `(S, 0)`, `(S, S)`,
    and `(0, S)`. Here `S = SIZE`.
    """
    return SIZE / 2


def uniform_on_circular_arc(
    center: Point,
    radius: float,
    arc_start_rad: float,
    arc_width_rad: float,
) -> Point:
    """Return a point uniformly distributed on a circular arc.

    Parameters
    ----------
    center : Point
        Center of the circle.
    radius : float
        Radius of the circle.
    arc_start_rad : float
        Starting angle (in radians) of the arc on which the generated point must lie.
    arc_width_rad : float
        Angular width (in radians) of the arc on which the generated point must lie.
        Must be in the range `(0, 2 * math.pi]`.
    """
    if arc_width_rad <= 0.0 or arc_width_rad > 2 * math.pi:
        raise ValueError(
            f"`arc_width_rad` must be in `(0, 2 * math.pi]`, but given "
            f"value was {arc_width_rad}"
        )
    angle_rad = random.uniform(arc_start_rad, (arc_start_rad + arc_width_rad))
    return center.translated(radius, angle_rad)


@dataclass(frozen=True)
class FirstBottomEdgeParams:
    """Parameters for bottom edge of first quadrilateral."""

    # Angular range for entry point w.r.t. circle center
    entry_point_arc_start_rad: float = 0.0
    entry_point_arc_width_rad: float = math.pi
    # Range for angle subtended at circle center by bottom edge of first quadrilateral
    first_bottom_edge_angle_min_rad: float = math.pi / 9
    first_bottom_edge_angle_max_rad: float = math.pi / 5


@dataclass(frozen=True)
class TopGivenBottomParams:
    """Parameters for generating top edge of quadrilateral from bottom edge."""

    # Range for distance along perpendicular bisector of bottom edge of a quadrilateral
    dist_along_perp_bisector_min: float = 45.0  # changed from 40.0
    dist_along_perp_bisector_max: float = 85.0  # changed from 80.0
    # Range for ratio of top edge length to bottom edge length
    top_to_bottom_ratio_min: float = 0.4  # changed from 0.5
    top_to_bottom_ratio_max: float = 0.6  # changed from 0.8
    # Range for angle by which to rotate top edge
    top_rot_angle_min_rad: float = -math.pi / 5  # changed from -math.pi / 9
    top_rot_angle_max_rad: float = math.pi / 5  # changed from math.pi / 9


def entry_point(
    entry_point_arc_start_rad: float, entry_point_arc_width_rad: float
) -> Point:
    """Return random entry point for tool image."""
    return uniform_on_circular_arc(
        center=circle_center(),
        radius=circle_radius(),
        arc_start_rad=entry_point_arc_start_rad,
        arc_width_rad=entry_point_arc_width_rad,
    )


def first_bottom_edge(params: FirstBottomEdgeParams) -> LineSegment:
    """Return random bottom edge of first quadrilateral."""
    start = entry_point(
        params.entry_point_arc_start_rad, params.entry_point_arc_width_rad
    )
    first_bottom_edge_angle_rad = random.uniform(
        params.first_bottom_edge_angle_min_rad, params.first_bottom_edge_angle_max_rad
    )
    end = start.rotated(angle_rad=first_bottom_edge_angle_rad, center=circle_center())
    return LineSegment(start, end)


def top_given_bottom(bottom: LineSegment, params: TopGivenBottomParams) -> LineSegment:
    """Return random top edge of a quadrilateral with given bottom edge."""
    dist_along_perp_bisector = random.uniform(
        params.dist_along_perp_bisector_min, params.dist_along_perp_bisector_max
    )
    top_to_bottom_ratio = random.uniform(
        params.top_to_bottom_ratio_min, params.top_to_bottom_ratio_max
    )
    top_rot_angle_rad = random.uniform(
        params.top_rot_angle_min_rad, params.top_rot_angle_max_rad
    )
    top_midpoint = bottom.point_on_perpendicular_bisector(dist_along_perp_bisector)
    return bottom.parallel(
        midpoint_of_parallel=top_midpoint,
        length_of_parallel=(top_to_bottom_ratio * bottom.length()),
    ).rotated(angle_rad=top_rot_angle_rad, center=top_midpoint)


def tip_circle_center_and_radius(top: LineSegment):
    """Draw a circle at the tip of the last quadrilateral."""
    center = Point((top.start.x + top.end.x) / 2, (top.start.y + top.end.y) / 2)
    radius = math.dist((top.start.x, top.start.y), (top.end.x, top.end.y)) / 2
    return center, radius


def draw_quadrilateral(
    draw: ImageDraw,
    bottom: LineSegment,
    top: LineSegment,
    fill_below_bottom_edge: bool = False,
) -> None:
    """Draw quadrilateral with given bottom and top edges."""
    fill_color = "white"
    draw.polygon(
        [
            (bottom.start.x, bottom.start.y),
            (bottom.end.x, bottom.end.y),
            (top.end.x, top.end.y),
            (top.start.x, top.start.y),
        ],
        fill=fill_color,
    )
    draw.line(
        [(bottom.start.x, bottom.start.y), (bottom.end.x, bottom.end.y)],
        fill=fill_color,
    )
    if fill_below_bottom_edge:
        # Fill circular segment formed by bottom edge
        start_angle_deg = LineSegment(circle_center(), bottom.start).angle() * 180 / math.pi
        end_angle_deg = LineSegment(circle_center(), bottom.end).angle() * 180 / math.pi
        draw.chord(
            [(0, 0), (SIZE, SIZE)],
            start=start_angle_deg,
            end=end_angle_deg,
            fill=fill_color,
        )


def find_vanishing_point(bottom: LineSegment, top: LineSegment):
    """Calculate the vanishing point based on the coordinates of the final trapezium."""
    x1 = bottom.start.x
    y1 = bottom.start.y
    x2 = bottom.end.x
    y2 = bottom.end.y
    x3 = top.end.x
    y3 = top.end.y
    x4 = top.start.x
    y4 = top.start.y

    left_slope = (y4 - y1) / (x4 - x1)
    right_slope = (y3 - y2) / (x3 - x2)

    left_c = y1 - left_slope * x1
    right_c = y2 - right_slope * x2

    vanishing_x = (left_c - right_c) / (right_slope - left_slope)
    vanishing_y = right_slope * vanishing_x + right_c

    return Point(vanishing_x, vanishing_y)


def create_vanishing_point_masks(root_path, vanishing_point: Point, i_image):
    """Creates groundtruth masks with the vanishing point"""
    mask = Image.new("RGB", (SIZE, SIZE))
    draw = ImageDraw.Draw(mask)
    draw.ellipse([(vanishing_point.x - 3, vanishing_point.y - 3),
                  (vanishing_point.x + 3, vanishing_point.y + 3)],
                  fill="white")
    mask.save(f"{root_path}/mask_{i_image}.jpg")


def create_vanishing_point_txt(root_path, vanishing_point: Point, i_image):
    """Create a text document with the vanishing point coordinates with respect
    to the image center: (v_x - c_x, v_y - c_y, 1)"""
    with open(f"{root_path}/vpoint_{i_image}.txt", 'w') as writefile:
        writefile.write(f"{vanishing_point.x - SIZE / 2} {vanishing_point.y - SIZE / 2} 1")


def draw_blob(img_path):
    # read input image
    img = cv2.imread(img_path)
    height, width = img.shape[:2]

    # define random seed to change the pattern
    seedval = random.randint(50, 80)
    rng = default_rng(seed=seedval)

    # create random noise image
    noise = rng.integers(0, 255, (height,width), np.uint8, True)

    # blur the noise image to control the size
    blur = cv2.GaussianBlur(noise, (0,0), sigmaX=15, sigmaY=15, borderType = cv2.BORDER_DEFAULT)

    # stretch the blurred image to full dynamic range
    stretch = skimage.exposure.rescale_intensity(blur, in_range='image', out_range=(0,255)).astype(np.uint8)

    # threshold stretched image to control the size
    lower_thresh = random.randint(140, 170)
    # thresh = cv2.threshold(stretch, 150, 255, cv2.THRESH_BINARY)[1]
    thresh = cv2.threshold(stretch, lower_thresh, 255, cv2.THRESH_BINARY)[1]

    # apply morphology open and close to smooth out and make 3 channels
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9,9))
    mask = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.merge([mask,mask,mask])

    # add mask to input
    result1 = cv2.add(img, mask)

    # use canny edge detection on mask
    edges = cv2.Canny(mask,50,255)

    # thicken edges and make 3 channel
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
    edges = cv2.morphologyEx(edges, cv2.MORPH_DILATE, kernel)
    edges = cv2.merge([edges,edges,edges])

    # merge edges with result1 (make black in result where edges are white)
    result2 = result1.copy()
    result2[np.where((mask == [255,255,255]).all(axis=2))] = [0,0,0]

    # save result
    cv2.imwrite(img_path, result2)
    # cv2_imshow(result2)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def create_tool_images_and_masks(root_path, num_images: int = 100, blob: bool = False):
    """Create and save given number of tool images."""
    for i_image in range(num_images):
        # Create image and draw circle
        img = Image.new("RGB", (SIZE, SIZE), color="black")
        draw = ImageDraw.Draw(img)
        draw.ellipse([(0, 0), (SIZE, SIZE)], outline="blue")

        # Draw first quadrilateral
        first_bottom = first_bottom_edge(FirstBottomEdgeParams())
        first_top = top_given_bottom(first_bottom, TopGivenBottomParams())
        draw_quadrilateral(draw, first_bottom, first_top, fill_below_bottom_edge=True)
        vanishing_point = find_vanishing_point(first_bottom, first_top)
        center, radius = tip_circle_center_and_radius(first_top)

        # Draw second quadrilateral (consider changing parameters here)
        second_bottom = first_top
        second_top = top_given_bottom(second_bottom, TopGivenBottomParams())
        draw_quadrilateral(draw, second_bottom, second_top, fill_below_bottom_edge=False)

        # Draw third quadrilateral (50% chance of choosing to draw third)
        if random.random() > 0.5:
            third_bottom = second_top
            third_top = top_given_bottom(third_bottom, TopGivenBottomParams())
            draw_quadrilateral(draw, third_bottom, third_top, fill_below_bottom_edge=False)
            # create mask for the vanishing point
            vanishing_point = find_vanishing_point(third_bottom, third_top)
            center, radius = tip_circle_center_and_radius(third_top)

        # if third quadrilateral is not created, use the second quadrilateral for the vanishing point
        else:
            vanishing_point = find_vanishing_point(second_bottom, second_top)
            center, radius = tip_circle_center_and_radius(second_top)

        # draw the circle at the tip of the last quadrilateral
        draw.ellipse([(center.x - radius + 1, center.y - radius + 1),
                      (center.x + radius - 1, center.y + radius - 1)],
                      fill="white")

        img.save(f"{root_path}/image_{i_image}.jpg")

        # draw blob if True
        if blob:
            draw_blob(f"{root_path}/image_{i_image}.jpg")

        # create_vanishing_point_masks(root_path, vanishing_point, i_image)
        create_vanishing_point_txt(root_path, vanishing_point, i_image)


if __name__ == "__main__":
    create_tool_images_and_masks("data/tool_vpoint_txt", 10000, blob=False)
