import matplotlib.pyplot as plt
import numpy as np
import random
from itertools import combinations
from time import time

def similar_radius(radii, rad=None, tol=0.05):
    if rad is None: # no target radius provided
        return max(radii) - min(radii) <= tol * max(radii)
    return all(abs(x - rad) <= (tol * rad) for x in radii)

def is_colinear(points, tol):
    pts = np.array(points, dtype=float)
    pts -= pts.mean(axis=0)
    _, s, _ = np.linalg.svd(pts)
    return s[1] / s[0] < tol

def is_uniform_spacing(points, spacing=None, tol=0.1):
    if spacing is None:
        sorted_pts = sorted(points, key=lambda p: (p[0], p[1]))
        dists = [np.linalg.norm(np.array(sorted_pts[i],dtype=float) - np.array(sorted_pts[i+1],dtype=float)) for i in range(len(sorted_pts) - 1)]
        mean_d = np.mean(dists)
        return all(abs(d - mean_d) < tol * mean_d for d in dists)
    # bit of a hack; sort by x then y (assumes things are roughly left to right...)
    sorted_pts = sorted(points, key=lambda p: (p[0], p[1]))
    dists = [np.linalg.norm(np.array(sorted_pts[i],dtype=float) - np.array(sorted_pts[i+1],dtype=float)) for i in range(len(sorted_pts) - 1)]
    return all(abs(d - spacing) < (tol * spacing) for d in dists)

# def is_valid_group(group, radius_tol, linear_tol, spacing_tol):
#     centers = [(x, y) for (x, y, _) in group]
#     radii = [r for (_, _, r) in group]
#     return similar_radius(radii, radius_tol) and is_colinear(centers, linear_tol) and is_uniform_spacing(centers, spacing_tol)

'''
def is_valid_group(group, radius,  radius_tol, spacing, spacing_tol, linear_tol):
    centers = [(x, y) for (x, y, _) in group]
    radii = [r for (_, _, r) in group]
    radius_ok = similar_radius(radii, radius, radius_tol) 
    spacing_ok = is_uniform_spacing(centers, spacing, spacing_tol)
    collinear_ok = is_colinear(centers, linear_tol)
    return radius_ok and spacing_ok and collinear_ok
'''

def is_valid_group(group, radius,  radius_tol, spacing, spacing_tol, linear_tol):
    centers = [(x, y) for (x, y, _) in group]
    radii = [r for (_, _, r) in group]
    radius_ok = similar_radius(radii, radius, radius_tol) 
    spacing_ok = is_uniform_spacing(centers, spacing, spacing_tol)
    collinear_ok = is_colinear(centers, linear_tol)
    #return radius_ok and spacing_ok and collinear_ok
    # Require radius always
    if not radius_ok:
        return False
    # Allow slight geometric imperfection:
    # Accept if either spacing OR collinearity holds
    if spacing_ok or collinear_ok:
        return True
    return False

def find_circle_groups(circles, radius, spacing, num_circles_in_group = 5, radius_tol=0.2, spacing_tol=0.1,  linear_tol=0.1):
    results = []
    start = time()
    # print("computing")
    for group in combinations(circles, num_circles_in_group):
        if is_valid_group(group, radius, radius_tol, spacing, spacing_tol, linear_tol):
            results.append(group)
    end = time()
    # print("done in", end - start, "sec")
    return results


# === Test Dataset Generation ===

def plot_circles(circles, title="Circles Visualization"):
    fig, ax = plt.subplots()
    for idx, (x, y, r) in enumerate(circles):
        circle = plt.Circle((x, y), r, fill=False, edgecolor='b')
        ax.add_patch(circle)
        ax.text(x, y, str(idx), fontsize=12, ha='center', va='center', color='r')
    min_dim = 0
    max_dim = 256
    # for idx, (x, y, r) in enumerate(circles):
    #     min_dim = min(min_dim, x - r)
    #     min_dim = min(min_dim, y - r)
    #     max_dim = max(max_dim, x + r)
    #     max_dim = max(max_dim, y + r)
    ax.set_aspect('equal', 'box')
    ax.set_xlim(min_dim, max_dim)
    ax.set_ylim(min_dim, max_dim)
    plt.title(title)
    plt.grid(True)
    plt.show()



# TESTING
TEST_CIRCLE_RADIUS = 7
TEST_CIRCLE_SPACING = 2.5*TEST_CIRCLE_RADIUS
TEST_CIRCLE_IMG_SIZE = 256
TEST_CIRCLE_ROW = 60
TEST_CIRCLE_RAD_VAR = 1
TEST_CIRCLE_LOC_VAR = 1
TEST_CIRCLE_LINEAR_VAR = 5  # degrees
def generate_test_data() -> np.ndarray:
    data = []
    img_size = TEST_CIRCLE_IMG_SIZE

    # Valid aligned group
    radius = TEST_CIRCLE_RADIUS
    spacing = TEST_CIRCLE_SPACING
    base_x, base_y = int(img_size/2-2*spacing), TEST_CIRCLE_ROW
    radius_range_px = TEST_CIRCLE_RAD_VAR
    loc_range_px = TEST_CIRCLE_LOC_VAR

    linear_skew = random.uniform(-TEST_CIRCLE_LINEAR_VAR, TEST_CIRCLE_LINEAR_VAR)
    for i in range(5):
        data.extend([
            (base_x + i * spacing + random.uniform(-loc_range_px, loc_range_px),
            base_y + random.uniform(-loc_range_px, loc_range_px) + (i * spacing * np.tan(np.deg2rad(linear_skew))), 
            radius + random.uniform(-radius_range_px, radius_range_px)),
        ])
    
    # Random noise circles
    # for _ in range(10):
    #     x = random.uniform(0, img_size)
    #     y = random.uniform(0, img_size)
    #     r = random.uniform(4, 20)
    #     data.append((x, y, r))
        
    return np.array(data)

def randomize_data(data):
    random.shuffle(data)
    return data


# === Run the Test ===

def main():
    test_circles = generate_test_data()
    # randomize_data(test_circles)

    # Find valid groups
    linear_fudge = 5
    radius_tol = np.ceil(TEST_CIRCLE_RAD_VAR/TEST_CIRCLE_RADIUS)
    linear_tol = linear_fudge*np.tan(np.deg2rad(TEST_CIRCLE_LINEAR_VAR))*5/TEST_CIRCLE_SPACING
    spacing_tol = np.ceil(TEST_CIRCLE_LOC_VAR/TEST_CIRCLE_SPACING)
    valid_groups = find_circle_groups(test_circles, 
                        radius = TEST_CIRCLE_RADIUS, spacing = TEST_CIRCLE_SPACING, 
                        radius_tol=radius_tol, linear_tol=linear_tol, spacing_tol=spacing_tol)

    # Print results
    if len(valid_groups) == 0:
        print("❌ No valid groups found.")
        plot_circles(test_circles, "Test Circles with Labels")
        return
    
    # circle_idx_map = {id(c): i for i, c in enumerate(test_circles)}
    # valid_group_indices = [[circle_idx_map[id(c)] for c in group] for group in valid_groups]
    # valid_group_indices = [[i for i, c in enumerate(test_circles) if c in group] for group in valid_groups]
    # if valid_group_indices:
    #     print("✅ Valid groups of 5 similar, evenly spaced, aligned circles:")
    #     for group in valid_group_indices:
    #         print("Group indices:", group)
    if valid_groups:
        print("✅ Valid groups of 5 similar, evenly spaced, aligned circles:")
        for group in valid_groups:
            print("Group centers:", group)
        plot_circles(test_circles, "Test Circles with Labels")

if __name__ == "__main__":
    for i in range(10):
        main()
