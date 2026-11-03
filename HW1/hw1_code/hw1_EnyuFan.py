#!/usr/bin/env python

##  hw1_skeleton.py

"""
This is the skeleton for Homework 1 of ECE 661, Computer Vision, Fall 2026.

The framework is written for you: reading the two supplied images, drawing the
overlays, saving the figures.  What is left for you to write is the geometry, and
every place that needs your work is marked TODO.

Everything in this homework is built out of four facts from Lecture 2:

    the line through two points            l  =  x1 cross x2
    the point where two lines meet         x  =  l1 cross l2
    a conic                                x^T C x  =  0
    the tangent to C at a point p on it    l  =  C p

Rename this file to hw1_<FirstName><LastName>.py before you submit it.  Note that
the work is all inside functions and that the calls at the bottom sit behind the
usual __main__ guard.  Keep it that way.  Your submission has to be safe to
import: a file that draws a window or reads an image at the moment it is imported
will fail on a machine that has neither a display nor the data directory.

    Usage:      python hw1_skeleton.py
    Requires:   numpy, matplotlib, pillow    (see the conda line in the handout)

Course policy on AI assistants applies to this file.  Your report must open with
a section titled "AI Assistant Usage" giving a percentage, and any function here
that was generated entirely by an AI assistant must say so in its own docstring.
"""

__author__ = "En-Yu Fan (fan474@purdue.edu)"
__date__   = "2026-Sep-02"

import os
import numpy as np
import matplotlib

##  Choose a drawing backend before pyplot is imported. 
if not (os.environ.get("DISPLAY") or os.name == "nt" or
        os.environ.get("MPLBACKEND")):
    matplotlib.use("Agg")

import matplotlib.pyplot as plt
from PIL import Image

"""
##  Uncomment this block if you want your recorded points and your figures to
##  come out the same on every run.  
seed = 0
random.seed(seed)
numpy.random.seed(seed)
"""


#______________________________  Lecture 2 in four functions  __________________________

##  These are given.  Read dehom() carefully: the comment inside it is the whole
##  reason we bother with homogeneous coordinates in the first place.

def hom(p):
    '''Physical point (x,y) as the homogeneous 3-vector [x, y, 1]^T.'''
    return np.array([p[0], p[1], 1.0])

def join(p, q):
    '''The line through the two physical points p and q.'''
    return np.cross(hom(p), hom(q))

def meet(l, m):
    '''The point where the two lines l and m cross.'''
    return np.cross(l, m)

def dehom(x, eps=1e-9):
    '''Homogeneous point back to physical, or None if the point is ideal.

    A homogeneous vector is defined only up to a nonzero scale, so asking
    whether "x3 is small" is meaningless on its own -- multiply the whole
    vector by a thousand and x3 grows with it.  The test has to be made
    relative to the other two entries, which is what is done below.

    Note also that [0, 0, c] with c nonzero is the ORIGIN and must come back
    as (0,0).  It is the zero vector that is not a point at all.
    '''
    if not np.any(x):
        return None
    scale = max(abs(x[0]), abs(x[1]))
    if abs(x[2]) <= eps * scale:
        return None
    return x[:2] / x[2]


#______________________________  Given framework  _______________________________________

##  Where the two supplied images live.  The default is the directory you get by
##  unzipping hw1_images.zip next to this file.  The environment variable is here
##  so that a grader can point the same code at a different directory without
##  editing it; you should not need to touch either.
DATA_DIR = os.environ.get("ECE661_HW1_DATA", "hw1_images")

def load_image(filename):
    '''Read an image from DATA_DIR as an (H, W, 3) array of uint8.'''
    path = os.path.join(DATA_DIR, filename)
    if not os.path.isfile(path):
        raise SystemExit(
            "Could not find %s.\n"
            "Unzip hw1_images.zip so that %s/ sits next to this file, or set the\n"
            "environment variable ECE661_HW1_DATA to the directory that has it."
            % (path, DATA_DIR))
    return np.array(Image.open(path).convert("RGB"))

def show_or_save(figure, filename):
    '''Put a figure on the screen if there is one, and always write it to disk.

    Every figure has to end up in your PDF report, so saving is not optional.
    Calling plt.show() with no display attached is an error, which is why this
    goes through the backend check made at the top of the file.
    '''
    figure.savefig(filename, dpi=110, bbox_inches="tight")
    if matplotlib.get_backend().lower() != "agg":
        plt.show()
    plt.close(figure)

def show_recorded_points(image, recorded, title):
    '''Draw the points you recorded on top of the image, and nothing else.

    Run this BEFORE you compute anything.  If a line you meant to record along
    a row of grout is drawn across the tiles instead, you will see it here.
    '''
    colors = ["#e4572e", "#2e86ab", "#3fa34d"]
    figure, axis = plt.subplots(figsize=(9, 7))
    axis.imshow(image)
    for (name, pairs), color in zip(recorded.items(), colors):
        for point_a, point_b in pairs:
            axis.plot([point_a[0], point_b[0]], [point_a[1], point_b[1]],
                      '-o', color=color, markersize=5, linewidth=1)
    axis.set_xlim(0, image.shape[1]);  axis.set_ylim(image.shape[0], 0)
    axis.set_title(title)
    return figure

def bounding_box_diagonal(points):
    '''The "spread" that Task-1(b) asks for: the diagonal of the bounding box.'''
    p = np.asarray(points, dtype=float)
    return float(np.hypot(np.ptp(p[:, 0]), np.ptp(p[:, 1])))

def draw_line_across(axis, line, xlim, color="#3fa34d", linewidth=2):
    '''Draw the line l = [a,b,c]^T across the given range of x.'''
    x = np.array(xlim, dtype=float)
    axis.plot(x, -(line[0] * x + line[2]) / line[1], '--',
              color=color, linewidth=linewidth)

def draw_conic(axis, C, image_shape, color="#e4572e", linewidth=2):
    '''Draw the curve x^T C x = 0 over the extent of an image.'''
    height, width = image_shape[:2]
    gx, gy = np.meshgrid(np.arange(0, width, 2.0), np.arange(0, height, 2.0))
    value = (C[0,0]*gx*gx + 2*C[0,1]*gx*gy + C[1,1]*gy*gy
             + 2*C[0,2]*gx + 2*C[1,2]*gy + C[2,2])
    axis.contour(gx, gy, value, levels=[0], colors=color, linewidths=linewidth)

def coefficients_to_matrix(c):
    '''The six coefficients [a,b,c,d,e,f] as the symmetric matrix of Lecture 2.'''
    a, b, cc, d, e, f = c
    return np.array([[a,   b/2, d/2],
                     [b/2, cc,  e/2],
                     [d/2, e/2, f  ]])

def two_panel_figure(image, recorded, vanishing_points, vanishing_line, title):
    '''The two-panel figure of Task-1(d).

    The near panel is scaled to the image, the wide panel to whatever vanishing
    points you actually computed, so this works on your own photograph in part
    (e) as well as on tiles.jpg.  If all of your vanishing points happen to
    land inside the frame, say so in your report and use one panel instead.
    '''
    height, width = image.shape[:2]
    vx = [v[0] for v in vanishing_points];  vy = [v[1] for v in vanishing_points]
    margin = 0.35 * max(width, height)
    wide_x = (min(-margin, min(vx) - margin), max(width + margin, max(vx) + margin))
    wide_y = (max(height + margin, max(vy) + margin), min(-margin, min(vy) - margin))
    #near_x = (-0.35 * width, 1.6 * width)
    #near_y = (1.15 * height, -0.4 * height)
    ##########################################################
    # make the view include image & two VPs nearest to center.
    img_center = np.array([width / 2.0, height / 2.0])
    near_vps = sorted(vanishing_points, 
                      key=lambda v: np.linalg.norm(v - img_center))[:2] # first 2

    near_vx = [v[0] for v in near_vps]
    near_vy = [v[1] for v in near_vps]
    # inclusive range for margin & most outside VPs.
    near_x = (min(-margin, min(near_vx) - margin),
              max(width + margin, max(near_vx) + margin))
    near_y = (max(height + margin, max(near_vy) + margin),
              min(-margin, min(near_vy) - margin))
    ##########################################################
    colors = ["#e4572e", "#2e86ab", "#3fa34d"]
    figure, axes = plt.subplots(1, 2, figsize=(14, 6.2))
    for axis, (xlim, ylim, subtitle) in zip(
            axes, [(near_x, near_y, "near view"), (wide_x, wide_y, "wide view")]):
        axis.imshow(image)
        for (name, pairs), color, v in zip(recorded.items(), colors, vanishing_points):
            for point_a, point_b in pairs:
                A = np.array(point_a, dtype=float);  B = np.array(point_b, dtype=float)
                #t = np.linspace(-1, 9, 2)
                #P = A + t[:, None] * (B - A)
                #axis.plot(P[:,0], P[:,1], color=color, linewidth=1.0, alpha=0.85)
                
                ##########################################################
                # Extend the line across the entire panel.
                axis.axline(A, B, color=color, linewidth=1.0, alpha=0.85)
                ##########################################################
                axis.plot([point_a[0], point_b[0]], [point_a[1], point_b[1]],
                          'o', color=color, markersize=4)
            if xlim[0] <= v[0] <= xlim[1] and ylim[1] <= v[1] <= ylim[0]:
                axis.plot(v[0], v[1], '*', color=color, markersize=17,
                          markeredgecolor='k', label="VP " + name.split()[0])
        draw_line_across(axis, vanishing_line,
                         (xlim[0] - 4000, xlim[1] + 4000), color='k')
        axis.set_xlim(*xlim);  axis.set_ylim(*ylim)
        ##########################################################
        #axis.legend(loc="lower right", fontsize=8)
        axis.legend(loc="upper right", fontsize=8) # to prevent overlapping VP A
        ##########################################################
        axis.set_title(subtitle, fontsize=9)
    figure.suptitle(title);  plt.tight_layout()
    return figure


#______________________________  Bonus (optional):  Shooting Around Cover  _____________

##  The arena, in an ordinary x-y plane with y upward.  The player is at the
##  origin and alpha is measured counter-clockwise from the +x axis.  No image
##  is involved in this section, and it is worth 10 bonus points.

TARGET = (np.array([1.0, 8.0]), np.array([3.0, 8.0]))
ARMOUR = (np.array([0.4, 4.0]), np.array([5.0, 4.0]))
WALL   = (np.array([7.0, 0.0]), np.array([7.0, 6.0]))

def draw_arena(axis):
    '''Given: the three segments and the player.'''
    for segment, color, label in [(TARGET, "#2e86ab", "target bar"),
                                  (ARMOUR, "#e4572e", "armour"),
                                  (WALL,   "#3fa34d", "mirror wall")]:
        axis.plot(*zip(*segment), color=color, linewidth=7,
                  solid_capstyle="butt", label=label)
    axis.plot(0, 0, 'ks', markersize=8)
    axis.set_aspect("equal");  axis.grid(alpha=0.25)
    axis.set_xlim(-1, 14.5);   axis.set_ylim(-1, 9.5)

def aiming_line(alpha_in_degrees):
    '''TODO (a): the aiming line, as a homogeneous 3-vector.

    It is the join of the origin and a point one unit away in the direction
    alpha.  Say in your report where alpha ends up in the three entries, and
    why the third entry comes out the way it does.
    '''
    # convert the input angle into radian for NumPy.
    alpha = np.deg2rad(alpha_in_degrees)

    # join player & a unit point in aiming direction.
    player = np.array([0.0, 0.0]) # origin
    direction_point = np.array([np.cos(alpha), np.sin(alpha)])
    return join(player, direction_point)

def subtended_interval(segment):
    '''TODO (a): the angular interval a segment subtends at the origin.

    Return the SMALLER of the two arcs bounded by the two endpoint directions.
    Watch the case where that arc straddles the zero direction.
    '''
    # find each endpoint direction
    angles = np.array([
        np.degrees(np.arctan2(point[1], point[0]))
        for point in segment
    ])
    # represent it in degree [0, 360)
    angles = np.mod(angles, 360.0)
    start, end = np.sort(angles)
    # If the direct interval > 180 degrees, use the arc across 0 to get the smaller arc
    if end - start > 180.0:
        return end, start + 360.0

    return start, end

def strikes_segment(segment, alpha_in_degrees, use_direction_test=True):
    '''TODO (b): does the shot at angle alpha strike this segment?

    Build the segment's line as the join of its endpoints, meet it with the
    aiming line, and decide whether the meet falls between the two endpoints.

    Write this FIRST with use_direction_test = False, run it at alpha = 250,
    and report what it claims about the target.  Then work out what is missing,
    add it under the flag, and run it again.
    '''
    """
        This implementation was written with assistance from an AI assistant.
    """
    # build 2 infinite lines and find the homogeneous intersection.
    segment_line = join(segment[0], segment[1])
    shot_line = aiming_line(alpha_in_degrees)
    q = dehom(meet(segment_line, shot_line))

    # parallel lines meet at an ideal point, not on a finite segment.
    if q is None:
        return False

    # the meet lies on the segment's line, so test whether it is within the bounds.
    tolerance = 1e-9 # in case some rounding error makes q fall just outside the segment
    lower_bound = np.minimum(segment[0], segment[1]) - tolerance
    upper_bound = np.maximum(segment[0], segment[1]) + tolerance
    # check if q's (x,y) lies within the bounding range
    falls_between = np.all(q >= lower_bound) and np.all(q <= upper_bound)

    if not falls_between:
        return False

    # check direction
    if use_direction_test:
        alpha = np.deg2rad(alpha_in_degrees)
        direction = np.array([np.cos(alpha), np.sin(alpha)])
        # reject an intersection that lies behind player
        if direction @ q <= 0.0:
            return False

    return True

def reflection_in_line(l):
    '''TODO (c): the reflection matrix M = I - 2 * n_tilde * l^T.

    The handout gives you the formula and the reason it is what it is.  This
    function should assume a^2 + b^2 = 1 and say so if that is not the case.
    '''
    # if a^2 + b^2 != 1.
    if not np.isclose(l[0] ** 2 + l[1] ** 2, 1.0):
        raise ValueError("Line normal should have unit length")

    n_tilde = np.array([l[0], l[1], 0.0])

    # the given reflection M's formula
    return np.eye(3) - 2.0 * np.outer(n_tilde, l) 

def bonus():
    ##  TODO (a): report both intervals, and show that for every angle in the
    ##            target's interval, the ray crosses y = 4 inside the armour.
    target_interval = subtended_interval(TARGET)
    armour_interval = subtended_interval(ARMOUR)
    print("\nBonus (a)")
    print(f"  Target interval: {target_interval[0]:.4f}, {target_interval[1]:.4f} degrees")
    print(f"  Armour interval: {armour_interval[0]:.4f}, {armour_interval[1]:.4f} degrees")
    if_contained = (
        target_interval[0] >= armour_interval[0]
        and target_interval[1] <= armour_interval[1]
    )
    print(f"  Target interval contained in armour interval: {if_contained}")

    # check where 2 boundary rays of the target interval cross y = 4.
    target_y = TARGET[0][1] # = 8
    armour_y = ARMOUR[0][1] # = 4
    boundary_angles = np.deg2rad(np.array(target_interval))
    crossing_x = armour_y / np.tan(boundary_angles)
    crossing_interval = (np.min(crossing_x), np.max(crossing_x)) 

    armour_x = sorted([ARMOUR[0][0], ARMOUR[1][0]]) # interval in order
    crossings_inside = (
        crossing_interval[0] >= armour_x[0]
        and crossing_interval[1] <= armour_x[1]
    ) # check if the ray crosses y = 4 inside the armour

    # the target rays travel "upward" through y = 4 before reaching y = 8.
    upward_rays = np.all(np.sin(boundary_angles) > 0.0)
    armour_met_first = (crossings_inside 
                        and upward_rays and 0.0 < armour_y < target_y
    )

    print(f"  Target rays cross y = {armour_y:.1f} "
          f"              over x = [{crossing_interval[0]:.4f}, {crossing_interval[1]:.4f}]")
    print(f"  Armour covers x = [{armour_x[0]:.4f}, {armour_x[1]:.4f}]")
    print(f"  Armour is met first: {armour_met_first}")
    print(f"  Direct shot exists: {not armour_met_first}")
    ##  TODO (a): meet the aiming line with l_infinity = [0,0,1]^T and say what
    ##            you get.  This is the first ideal point you will compute.
    sample_alpha = target_interval[0]
    sample_line = aiming_line(sample_alpha)
    line_at_infinity = np.array([0.0, 0.0, 1.0])
    ideal_point = meet(sample_line, line_at_infinity)

    # Normalize the first two entries to print clearer
    ideal_point = ideal_point / np.linalg.norm(ideal_point[:2])
    print(f"\nSample alpha: {sample_alpha:.4f} degrees")
    print(f"  Aiming line: {sample_line}")
    print(f"  Ideal point: {ideal_point}")
    ##  TODO (b): the naive test at alpha = 250, then the corrected one.
    test_alpha = 250.0
    # the requested naive test with undirected aiming line.
    naive_hit = strikes_segment(TARGET, test_alpha, use_direction_test=False)

    # recompute the meet to report its position and direction test
    target_line = join(TARGET[0], TARGET[1])
    q = dehom(meet(target_line, aiming_line(test_alpha)))

    alpha = np.deg2rad(test_alpha)
    direction = np.array([np.cos(alpha), np.sin(alpha)])
    direction_test = direction @ q

    # the actual test
    corrected_hit = strikes_segment(TARGET, test_alpha, use_direction_test=True)
    print("\nBonus (b)")
    print(f"  Meet q: ({q[0]:.4f}, {q[1]:.4f})")
    print(f"  Naive line test reports a hit: {naive_hit}")
    print(f"  Direction d: ({direction[0]:.4f}, {direction[1]:.4f})")
    print(f"  d^T q: {direction_test:.4f}")
    print(f"  Corrected ray test reports a hit: {corrected_hit}")
    ##  TODO (c): M for the wall x = 7; check that M @ M is the identity.
    wall_line = np.array([1.0, 0.0, -7.0])# x = 7
    reflection_M = reflection_in_line(wall_line)
    reflection_squared = reflection_M @ reflection_M
    is_I = np.allclose(reflection_squared, np.eye(3))

    print("\nBonus (c)")
    print(f"  Wall line l: {wall_line}")
    print("  Reflection matrix M:")
    print(reflection_M)
    print("  M^2:")
    print(reflection_squared)
    print(f"  M^2 equals I: {is_I}")
    ##  TODO (d): reflect the target bar in the wall, aim at the mirror image,
    ##            then trace the ray and check every leg of it.  Plot the path
    ##            on top of draw_arena().

    # reflect both target endpoints across the wall.
    reflect_target = tuple(dehom(reflection_M @ hom(point))
                             for point in TARGET)
    mirror_center = (reflect_target[0] + reflect_target[1]) / 2.0

    # aim at mirror & compute angle
    # aim from player(0, 0) at the center of the reflected target & get shooting line.
    ricochet_alpha = np.degrees(np.arctan2(mirror_center[1], mirror_center[0]))# (y, x)
    shoot_line = aiming_line(ricochet_alpha)

    # confirm that it clears the armour.
    # find where the incoming path crosses armour line.
    arm_line = join(ARMOUR[0], ARMOUR[1])
    arm_crossing = dehom(meet(shoot_line, arm_line)) # intersect y=4
    clear_arm = not strikes_segment(ARMOUR, ricochet_alpha)

    # find where the incoming path strikes the finite wall segment.
    wall_hit = dehom(meet(shoot_line, wall_line))
    hits_wall = strikes_segment(WALL, ricochet_alpha)

    # use ideal point from (a) to represent the aiming direction.
    aiming_ideal = meet(shoot_line, line_at_infinity)
    aiming_ideal = (aiming_ideal / np.linalg.norm(aiming_ideal[:2]))

    # reflect direction
    # multiply reflection M with incoming direction as a homogeneous vec
    reflected_ideal = reflection_M @ aiming_ideal
    reflected_direction = (reflected_ideal[:2] / np.linalg.norm(reflected_ideal[:2]))

    # return line from the wall hit and reflected direction.
    return_line = join(wall_hit,  wall_hit + reflected_direction)
    target_line = join(TARGET[0], TARGET[1])
    landing_point = dehom(meet(return_line, target_line))

    target_x = sorted([TARGET[0][0], TARGET[1][0]])
    land_on_target = (target_x[0] <= landing_point[0] <= target_x[1])

    print("\nBonus (d)")
    print(f"  Reflected target endpoints: "
          f"({reflect_target[0][0]:.4f}, "
          f"{reflect_target[0][1]:.4f}) and "
          f"({reflect_target[1][0]:.4f}, "
          f"{reflect_target[1][1]:.4f})"
    )
    print(f"  Reflected target center: "
          f"({mirror_center[0]:.4f}, {mirror_center[1]:.4f})"
    )
    print(f"  Aiming angle alpha: {ricochet_alpha:.4f} degrees")
    print(f"  Incoming ideal point: "
          f"[{aiming_ideal[0]:.4f}, "
          f"{aiming_ideal[1]:.4f}, "
          f"{aiming_ideal[2]:.4f}]"
    )
    print(f"  Crossing at armour height: "
          f"({arm_crossing[0]:.4f}, "
          f"{arm_crossing[1]:.4f})"
    )
    print(f"  Clears armour: {clear_arm}")
    print(f"  Wall hit: "
          f"({wall_hit[0]:.4f}, {wall_hit[1]:.4f})"
    )
    print(f"  Hits wall segment: {hits_wall}")
    print(f"  Reflected ideal point: "
          f"[{reflected_ideal[0]:.4f}, "
          f"{reflected_ideal[1]:.4f}, "
          f"{reflected_ideal[2]:.4f}]"
    )
    print(f"  Reflected direction: "
          f"({reflected_direction[0]:.4f}, "
          f"{reflected_direction[1]:.4f})"
    )
    print(f"  Return-leg landing point: "
          f"({landing_point[0]:.4f}, "
          f"{landing_point[1]:.4f})"
    )
    print(f"  Return leg lands on target: {land_on_target}")

    ##########
    figure, axis = plt.subplots(figsize=(8, 6))
    draw_arena(axis)

    # Show the reflected target and the straight mirror construction.
    axis.plot([reflect_target[0][0], reflect_target[1][0]],
              [reflect_target[0][1], reflect_target[1][1]],
              "--", color="#2e86ab", linewidth=3,
              label="reflected target"
    )
    axis.plot([0.0, mirror_center[0]],
              [0.0, mirror_center[1]],
              "--", color="gray", linewidth=1.5,
              label="mirror construction"
    )

    # Draw the actual two-leg path through the wall hit.
    axis.plot([0.0, wall_hit[0], landing_point[0]],
              [0.0, wall_hit[1], landing_point[1]],
              color="#8e44ad", linewidth=2.5, marker="o",
              label="two-leg path"
    )

    axis.set_title("Bonus (d)")
    axis.legend(loc="upper right", fontsize=8)
    plt.tight_layout()
    show_or_save(figure, "Bonus(d)_ricochet_path.png")


#______________________________  Task-1:  Vanishing Points and the Vanishing Line  _____

##  TODO (a): replace the zeros with the coordinates YOU recorded in GIMP.
##
##  Families A and B run along grout lines.  Family C is the tile diagonals,
##  which are not drawn in the image: record a corner where four tiles meet,
##  then step diagonally across the grid and record another such corner.  Count
##  the tiles as you go.  A miscount gives you a line that is not in the family
##  and the image will not tell you so -- but show_recorded_points() will.

RECORDED_TILES = {
    "A  tile rows":      [((297, 115), (1281, 509)),
                          ((270, 358), (1496, 811)),
                          ((99, 514), (1304, 927))],
    "B  tile columns":   [((549, 215), (47, 942)),
                          ((711, 233), (621, 1107)),
                          ((901, 298), (1268, 1132))],
    "C  tile diagonals": [((364, 161), (1387, 884)),
                          ((173, 274), (1071, 1070)),
                          ((60, 342), (881, 1182))],
}


RECORDED_OWN = {
    "A  tile rows":      [((1436, 605), (2847, 966)),
                          ((1091, 816), (2801, 1389)),
                          ((908, 1305), (2290, 1972))],
    "B  tile columns":   [((2243, 476), (815, 1036)),
                          ((2603, 595), (1126, 1413)),
                          ((3088, 763), (1921, 1793))],
    "C  tile diagonals": [((2305, 536), (3098, 1495)),
                          ((1823, 554), (2147, 1590)),
                          ((1269, 651), (1058, 1703))],
}

def vanishing_point_estimates(pairs_of_points):
    '''TODO (b): the vanishing point of one family, from all three line-pairs.

    Build the three lines as joins, then take all three pairs of those lines
    and meet each pair.  Return the three estimates.  They will not agree.

    Return their MEAN as well, and use the mean everywhere in (c) and (d).
    Which single pair you would otherwise pick is not a neutral choice: on
    tiles.jpg the residual of part (c) moves by a factor of several hundred
    across the possible choices.  The mean is the estimate this assignment
    marks against.
    '''
    # build homogeneous lines from n pairs of points
    lines = [join(p, q) for p, q in pairs_of_points]

    # intersects the 3 pairs & dehomogenize each intersection -> 3 vanishing point estimates
    estimates = []
    for i, j in [(0, 1), (0, 2), (1, 2)]:
        vp = dehom(meet(lines[i], lines[j]))
        if vp is None:  # in cae of choosing an unsuitable img
            raise ValueError(f"Line {i + 1} and {j + 1} have no finite VP")
        estimates.append(vp)

    estimates = np.asarray(estimates, dtype=float)
    mean_vp = np.mean(estimates, axis=0)

    return estimates, mean_vp

def task_1(img_name, recorded, output_task, img_c):
    tiles = load_image(img_name)
    ##  Look before you compute.
    figure = show_recorded_points(tiles, recorded,
                                  "do these lie along the features you meant?")
    show_or_save(figure, f"{output_task}_recorded_points.png")

    ##  TODO (b): WRITE YOUR PREDICTION IN YOUR REPORT BEFORE RUNNING THIS.
    ##            Which family will give the least reliable vanishing point,
    ##            and why?  Then compute the three estimates per family, their
    ##            spread, each family's MEAN vanishing point, and its distance
    ##            from the image centre (800, 600), and say whether the
    ##            prediction held.
    res = {}
    img_center = img_c
    print("\nTask 1(b)")
    for name, pairs in recorded.items():
        # VP estimates, spread, and distance to the image center.
        estimates, mean_vp = vanishing_point_estimates(pairs)
        spread = bounding_box_diagonal(estimates)
        dist = float(np.linalg.norm(mean_vp - img_center))

        # store results for later parts
        res[name] = {
            "estimates": estimates,
            "spread": spread,
            "mean_vp": mean_vp,
            "distance": dist
        }
        # task 1-b output
        print(f"\n{name}")
        for (i, j), vp in zip([(1, 2), (1, 3), (2, 3)], estimates):
            print(f"VP-{i}{j}: ({vp[0]:.2f}, {vp[1]:.2f})")

        print(f"  Spread: {spread:.2f} pixels")
        print(f"  Mean VP: ({mean_vp[0]:.2f}, {mean_vp[1]:.2f})")
        print(f"  Distance to center: {dist:.2f} pixels")
    # check if prediction held
    worst_one = max(res, key=lambda name: res[name]["spread"])
    print(f"\nLargest spread: {worst_one}")
    ##  TODO (c): order the families by spread; compute the perpendicular
    ##            distance from the WORST-determined mean vanishing point to
    ##            the line through the other two mean vanishing points.
    ordered_family = sorted(res, key=lambda name: res[name]["spread"])
    good_1, good_2, worst_one = ordered_family

    # use mean VPs from 1(b).
    vp_1 = res[good_1]["mean_vp"]
    vp_2 = res[good_2]["mean_vp"]
    vp_worst = res[worst_one]["mean_vp"]
    # line through 2 better VPs
    vanishing_line = join(vp_1, vp_2)

    # perpendicular distance from the worst VP to this line, ax+by+c=0
    line_norm = np.linalg.norm(vanishing_line[:2])
    # just in case accidentlly enter 2 same families
    if line_norm == 0:
        raise ValueError("2 better families's VPS not defining a valid finite line")
    # dist between point & the line
    residual = float(abs(np.dot(vanishing_line, hom(vp_worst))) / line_norm)

    print("\nTask 1(c)")
    for name in ordered_family:
        print(f"  {name}: {res[name]['spread']:.2f} pixels")

    print(f"Line goes through: {good_1} & {good_2}")
    print(f"Collinearity residual: {residual:.2f} pixels")

    ##  TODO (d): the vanishing line as the join of the two BEST-determined
    ##            vanishing points, then two_panel_figure().

    # use original family order -> colors match the recorded points.
    mean_vps = [res[name]["mean_vp"] for name in recorded]

    # normalize the vanishing line computed in 1(c)
    a, b, c = vanishing_line / line_norm

    print("\nTask 1(d)")
    print(f"  [a:{a:.4f}, b:{b:.4f}, c:{c:.4f}]")

    figure = two_panel_figure(
        tiles, recorded,
        mean_vps, vanishing_line,
        output_task
    )
    show_or_save(figure, f"{output_task}_vline.png")

    ##  TODO (e): the same on the photograph you took yourself.
    # access this task1() with my own recorded points and image data, in main()


#______________________________  Task-2:  A Circle Photographed From an Angle  _________

##  TODO (a): replace with the eight points YOU recorded around the OUTER edge
##            of the plate.  Do not mix in the inner ring where the glaze
##            changes shade; that is a second, smaller circle.

RECORDED_RIM = np.array([[488, 651], [613, 508], [800, 430], [1021, 426],
                         [1076, 616], [939, 721], [752, 780], [586, 777]], dtype=float)

                         
def fit_conic(points):
    '''TODO (b): fit a conic to a set of points.
    Each point contributes one row [x^2, xy, y^2, x, y, 1] to a design matrix.
    Since C is fixed only up to scale, what you want is the null direction of
    that matrix: the right singular vector belonging to its SMALLEST singular
    value.  numpy.linalg.svd returns U, s and Vt, and the vector you want is
    the last ROW of Vt.

    Return C, scaled so that its Frobenius norm is 1.  Fixing the scale is not
    optional -- x^T C x is linear in C, so without a convention the residual
    you report in part (b) could be made any number at all.

    (Keep the shape of this in mind.  Building a design matrix whose null
    vector is the answer, because the answer is only defined up to scale, is
    exactly how you will estimate a homography in HW2.)
    '''
    """
        This implementation was written with assistance from an AI assistant.
    """
    points = np.asarray(points, dtype=float)
    x = points[:, 0]
    y = points[:, 1]

    # each point contributes one row to the conic design matrix.
    # shape: (8, 6)
    A = np.column_stack((x**2, x*y, y**2, x, y, np.ones_like(x)))

    # coef. vec = "right singular vector" in smallest singular value of A
    # which is the last row of Vt
    _, _, Vt = np.linalg.svd(A, full_matrices=False)
    coefficients = Vt[-1]

    # convert 6 coefficients into symmetric conic matrix.
    # shape: (3, 3)
    conic_matrix = coefficients_to_matrix(coefficients)

    # fix the otherwise arbitrary scale of the conic matrix.
    frobenius_norm = np.linalg.norm(conic_matrix, ord="fro")
    if frobenius_norm == 0:
        raise ValueError("conic fit produced a zero matrix")

    return conic_matrix / frobenius_norm # C with Frobenius norm 1


def task_2():
    plate = load_image("plate.jpg")

    figure, axis = plt.subplots(figsize=(8, 6))
    axis.imshow(plate)
    axis.plot(RECORDED_RIM[:,0], RECORDED_RIM[:,1], 'o',
              color="#2e86ab", markersize=7, markeredgecolor='k')
    axis.set_xlim(0, plate.shape[1]);  axis.set_ylim(plate.shape[0], 0)
    axis.set_title("your recorded rim points")
    show_or_save(figure, "task2_recorded_points.png")

    ##  TODO (b): fit C and report max_i |x_i^T C x_i| with ||C||_F = 1.
    ##            Expect something of order 1e-3 to 1e-4.  If you get order 1,
    ##            you have taken the wrong singular vector.
    recorded_conic = fit_conic(RECORDED_RIM)

    # evaluate the normalized conic equation at all eight recorded points.
    residuals = [
        abs(hom(point) @ recorded_conic @ hom(point)) # x^T C x
        for point in RECORDED_RIM
    ]
    max_residual = max(residuals)

    print("\nTask 2(b)")
    print(recorded_conic)
    print(f"Maximum algebraic residual: {max_residual:.6e}")
    ##  TODO (c): rescale so that a^2 + b^2 + c^2 = 1, report b^2 - 4ac and
    ##            classify.  Expect a value near -0.86 on plate.jpg.
    # Recover [a, b, c, d, e, f] from the symmetric conic matrix.
    coeffs = np.array([
        recorded_conic[0, 0], 2 * recorded_conic[0, 1],
        recorded_conic[1, 1], 2 * recorded_conic[0, 2],
        2 * recorded_conic[1, 2], recorded_conic[2, 2]
    ], dtype=float)

    # rescale using only the 3 quadratic coefficients (a,b,c) so that a^2 + b^2 + c^2 = 1
    abc_norm = np.linalg.norm(coeffs[:3])
    scaled_coeffs = coeffs / abc_norm
    a, b, c = scaled_coeffs[:3]

    # classify the conic by the discriminant.
    discriminant = b**2 - 4 * a * c

    if np.isclose(discriminant, 0.0, atol=1e-10):
        conic_type = "parabola"
    elif discriminant < 0:
        conic_type = "ellipse"
    else:
        conic_type = "hyperbola"

    print("\nTask 2(c)")
    print(f"  a = {a:.4f}, b = {b:.4f}, c = {c:.4f}")
    print(f"  a^2 + b^2 + c^2 = {a**2 + b**2 + c**2:.4f}")
    print(f"  Discriminant = {discriminant:.4f}")
    print(f"  Conic Type: {conic_type}")
    ##  TODO (d): the tangent at one recorded point, l = C p.  Overlay the
    ##            conic, the points and the tangent, then zoom in on p and
    ##            confirm the line touches the curve there without crossing it.
    # choose 1 recorded rim point & compute its tangent line l = C p.
    p = RECORDED_RIM[2] # 3rd point (800, 430), since it's relatively near horizontal
    p_hom = hom(p)
    tg_line = recorded_conic @ p_hom

    # overlay the fitted conic, recorded points, and tangent line.
    figure, axes = plt.subplots(1, 2, figsize=(14, 6))
    height, width = plate.shape[:2]
    zoom_radius = 80

    for axis in axes:
        axis.imshow(plate)

        # draw the fitted conic and 8 recorded points.
        draw_conic(axis, recorded_conic, plate.shape)
        axis.plot(
            RECORDED_RIM[:, 0], RECORDED_RIM[:, 1], "o",
            color="#2e86ab", markersize=6, markeredgecolor="k"
        )

        # Mark the selected point and draw its tangent.
        axis.plot(
            p[0], p[1], "*",
            color="#f4c542", markersize=14, markeredgecolor="k"
        )
        draw_line_across(
            axis, tg_line, (0, width),
            color="#3fa34d", linewidth=2
        )

        axis.set_aspect("equal")

    # image view.
    axes[0].set_xlim(0, width)
    axes[0].set_ylim(height, 0)
    axes[0].set_title("Full overlay")
    # closer view around chosen point.
    axes[1].set_xlim(p[0] - zoom_radius, p[0] + zoom_radius)
    axes[1].set_ylim(p[1] + zoom_radius, p[1] - zoom_radius)
    axes[1].set_title("Tangent near chosen point")

    figure.suptitle("Task 2(d)")
    plt.tight_layout()
    show_or_save(figure, "task2_tangent.png")

    # conic equation & line equation at chosen point.
    pCp = float(p_hom @ recorded_conic @ p_hom)
    line_at_p = float(tg_line @ p_hom)

    print("\nTask 2(d): tangent check")
    print(f"  Point p = ({p[0]:.0f}, {p[1]:.0f})")
    print(f"  p^T C p = {pCp:.6e}")
    print(f"  l^T p   = {line_at_p:.6e}")


#______________________________  main  _________________________________________________

if __name__ == '__main__':
    
    task_1(
        "tiles.jpg",
        RECORDED_TILES,
        "task1",
        np.array([800.0, 600.0])
    )

    print("################################################")
    print("################# Next 1(e) ####################")
    print("################################################")

    task_1(
        "bricks.jpg",
        RECORDED_OWN,
        "task1_e",
        np.array([2016.0, 1134.0])
    )

    print("################################################")
    print("################### Next 2 #####################")
    print("################################################")

    task_2()
    
    print("################################################")
    print("#################### Bonus #####################")
    print("################################################")

    bonus()
