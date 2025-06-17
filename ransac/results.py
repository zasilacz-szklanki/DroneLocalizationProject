import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

recordings = [
    "wall1",
    "wall2",
    "wall3"
]
algorithms =["BeamformerBase","BeamformerOrth","BeamformerCleansc","BeamformerCMF"]

scale = 550
range_filter = True
outlier_filter = True

with open(Path("./dist.log"), "w") as f:
    f.write("")

def affine_transformation_matrix(src_points, dst_points):
    # Ensure the points are in the correct shape
    assert src_points.shape == dst_points.shape
    n = src_points.shape[0]

    # Add a column of ones to the source points for the affine transformation
    src_points_h = np.hstack([src_points, np.ones((n, 1))])

    # Solve the least squares problem to find the transformation matrix
    transformation_matrix, _, _, _ = np.linalg.lstsq(src_points_h, dst_points, rcond=None)

    return transformation_matrix.T # Transpose the result for correct indexing

def separate_linear_affine(matrix):
    # Extract the linear part (2x2 matrix) and affine part (translation vector)
    linear_part = matrix[:2, :2]
    affine_part = matrix[:2, 2]
    return linear_part, affine_part

def transform_points(src_points, linear_part, affine_part):
    # Multiply the source points by the linear part
    transformed_points = np.dot(src_points, linear_part)
    # Add the affine part (translation)
    transformed_points += affine_part
    return transformed_points

def ransac_affine(src_pts, dst_pts, max_trials=1000, residual_threshold=1.0):

    n_samples = src_pts.shape[0]
    best_inliers = []
    best_affine_matrix = None

    for _ in range(max_trials):
        sample_indices = np.random.choice(n_samples, 3, replace=False)
        src_sample = src_pts[sample_indices]
        dst_sample = dst_pts[sample_indices]

        affine_matrix = affine_transformation_matrix(src_sample, dst_sample)
        linear_part, affine_part = separate_linear_affine(affine_matrix)
        transformed_pts = transform_points(src_pts, linear_part, affine_part)

        residuals = np.linalg.norm(transformed_pts - dst_pts, axis=1)
        inliers = np.where(residuals < residual_threshold)[0]

        if len(inliers) > len(best_inliers):
            best_inliers = inliers
            best_affine_matrix = affine_matrix

    return best_affine_matrix


for rec in recordings:
    trajectory_file = Path(f"../matlab/avg/{rec}_avg.npy")
    trajectory = np.load(trajectory_file)

    print(f"\n{rec} shape: {trajectory.shape}")

    trajectory_indices = np.array([i for i, el in enumerate(trajectory) if not np.all(np.isnan(el))])
    trajectory = trajectory[trajectory_indices]

    for algo in algorithms:
        print(f"\nProcessing {rec} with {algo}")
        focuspoints_file = Path(f"../output_final/{rec}_{algo}_focuspoints.npy")
        focuspoints = np.load(focuspoints_file)
        print(focuspoints.shape)
        focuspoints = focuspoints[trajectory_indices]

        fp_size = focuspoints.shape[0]
        tr_size = trajectory.shape[0]

        last_index = min(fp_size, tr_size)

        src_points = focuspoints[:last_index]
        dst_points = trajectory[:last_index]

        print(f"Focuspoints shape: {focuspoints.shape}")
        print(f"Trajectory shape: {trajectory.shape}")

        print("Source points contain NaN: ", np.isnan(src_points).any())
        print("Destination points contain NaN: ", np.isnan(dst_points).any())

        # Oblicz macierz przekształcenia za pomocą RANSAC, jeśli nie istnieje
        matrix_file = Path(f"./matrix/{rec}_affine_matrix.npy")

        print(f"Calculating affine transformation matrix for {rec} using RANSAC")
        matrix = ransac_affine(src_points, dst_points, max_trials=1000, residual_threshold=5.0)
        np.save(matrix_file, matrix)
        print(f"Matrix saved to {matrix_file}")

        linear_part, affine_part = separate_linear_affine(matrix)

        focuspoints = transform_points(focuspoints, linear_part, affine_part)

        if range_filter:
            focuspoints[focuspoints < (654, 0)] = np.nan
            focuspoints[focuspoints > (1254, 1920)] = np.nan

        if outlier_filter:
            focuspoints_diff = np.abs(np.diff(focuspoints, axis=0, append=0))
            focuspoints[focuspoints_diff > 30] = np.nan

        # Oblicz błąd
        dist = np.linalg.norm(focuspoints[:last_index] - trajectory[:last_index], ord=np.inf, axis=1)

        dist_filt = []
        dist_diff = np.ediff1d(dist, to_begin=0)
        for i in range(1, dist_diff.shape[0]):
            d = dist[i]
            if np.isnan(d):
                continue
            dist_filt.append(d)

        dist = np.array(dist_filt)

        # Zapisz błąd
        np.save(f"./dist/{rec}_{algo}_dist_focuspoints.npy", dist)

        # Wyświetl statystyki
        avg_dist = np.average(dist)
        med_dist = np.median(dist)
        std_dist = np.std(dist)
        rmse = np.sqrt(np.mean(dist * dist))

        print("Average distance: ", avg_dist)
        print("Median distance: ", med_dist)
        print("Standard deviation: ", std_dist)
        print("RMSE: ", rmse)

        with open(Path("./dist.log"), "a") as f:
            f.write(f"{rec},{algo},{avg_dist},{med_dist},{std_dist},{rmse}\n")

        fix, ax = plt.subplots()
        x = np.arange(len(dist))
        ax.stem(x, dist, markerfmt='', label='Distance')
        ax.stem(x, np.abs(np.ediff1d(dist, to_begin=0)), markerfmt='', linefmt='C1-', label='Distance Difference')
        ax.set_ylim(0, 100)
        ax.set_xlabel("Sample Index")
        ax.set_ylabel("Distance [px]")
        ax.legend()
        ax.grid()
        plt.title(f"Error Distribution - {rec} with {algo}")
        plt.savefig(f"./error/{rec}_{algo}_error_dist.svg", format='svg')
        plt.close()

        # Wykres porównawczy
        fig, ax = plt.subplots()
        z = np.linspace(0, 1, focuspoints.shape[0])
        ax.scatter(focuspoints[:, 0], focuspoints[:, 1], c=z, s=0.2, cmap='autumn', label=f'{algo} Focuspoints')
        z = np.linspace(0, 1, trajectory.shape[0])
        ax.scatter(trajectory[:, 0], trajectory[:, 1], c=z, s=0.2, cmap='winter', label='Trajectory')
        ax.set_xlim(0, 1920)
        ax.set_ylim(0, 1080)
        ax.yaxis.set_inverted(True)
        ax.set_aspect("equal")
        ax.legend()
        plt.title(f"{rec} - {algo}")
        plt.savefig(f"./cmp/{rec}_{algo}_comparison.svg", format='svg')
        plt.close()
