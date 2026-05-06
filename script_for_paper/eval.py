import matplotlib
matplotlib.use('TkAgg')
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from utils import utils
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


kVizParams = {
    'dro': {'color': 'orange', 'label': 'DRO'},
    'pogo': {'color': 'blue', 'label': 'DR-PoGO'},
    'gt': {'color': 'red', 'label': 'Groundtruth'},
    'loop': {'color': 'green', 'label': 'Loop closure'}
}



def main():
    # Get the folders in the output directory
    output_paths = os.listdir("output")

    # Sort the output paths
    output_paths.sort()

    errors_per_type = {}
    epe_errors_per_type = {}


    for seq_id in output_paths:
        if not seq_id.startswith('boreas-'):
            continue

        print(f"Processing sequence {seq_id}...")

        # Read the loops
        try:
            loops = pd.read_csv(os.path.join("output", seq_id, "coarse_registrations.csv"))
        except Exception as e:
            loops = None


        # Get the GT, odom and pogo radar poses
        odom_poses, odom_times = utils.getDroPosesAndTimes(seq_id)
        pogo_poses, pogo_times = utils.getPogoPosesAndTimes(seq_id)
        if(np.any(odom_times != pogo_times)):
            raise ValueError("Odom and Pogo times do not match!")

        # Convert times to seconds
        odom_times = odom_times * 1e-6
        pogo_times = pogo_times * 1e-6
        

        compute_gt = True
        gt_interp_path = os.path.join("output", seq_id, "gt_interpolated.npz")
        if(os.path.exists(gt_interp_path)):
            data = np.load(gt_interp_path)
            gt_poses_interp = data['poses']
            gt_times = data['times']
            if(np.max(np.abs(gt_times - odom_times))) > 1e-3:
                compute_gt = True
            else:
                compute_gt = False

        if compute_gt:
            gt_poses, gt_times = utils.getGTRadarPosesAndTimes(seq_id)
            gt_poses_interp = utils.getInterpolatedTrajectory(gt_poses, gt_times, odom_times)
            inv_gt_first = np.linalg.inv(gt_poses_interp[0]).reshape(1,4,4)
            gt_poses_interp = inv_gt_first @ gt_poses_interp

            np.savez(gt_interp_path, poses=gt_poses_interp, times=odom_times)


        # Align the poses with identity
        inv_odom_first = np.linalg.inv(odom_poses[0]).reshape(1,4,4)
        inv_pogo_first = np.linalg.inv(pogo_poses[0]).reshape(1,4,4)

        odom_poses = inv_odom_first @ odom_poses
        pogo_poses = inv_pogo_first @ pogo_poses

        # Compute the absolute trajectory errors
        ate_odom = utils.get2dATE(
                gt_poses_interp
                , odom_poses
                , save_fig=True
                , est_colour=kVizParams['dro']['color']
                , est_label=kVizParams['dro']['label']
                , gt_colour=kVizParams['gt']['color']
                , gt_label=kVizParams['gt']['label']
                , path=os.path.join("output", seq_id, seq_id+"_odom_aligned.pdf")
                )
        ate_pogo = utils.get2dATE(
                gt_poses_interp
                , pogo_poses
                , save_fig=True
                , est_colour=kVizParams['pogo']['color']
                , est_label=kVizParams['pogo']['label']
                , gt_colour=kVizParams['gt']['color']
                , gt_label=kVizParams['gt']['label']
                , path=os.path.join("output", seq_id, seq_id+"_pogo_aligned.pdf")
                )

        print("2D Absolute Trajectory Error (RMSE ATE), odom:", ate_odom, "m, pogo:", ate_pogo, "m")

        # Store the errors
        seq_type = utils.getSeqType(seq_id)
        if seq_type not in errors_per_type:
            errors_per_type[seq_type] = {'odom': [], 'pogo': [], 'seq_id': []}
            epe_errors_per_type[seq_type] = {'odom': [], 'pogo': [], 'seq_id': []}
        errors_per_type[seq_type]['odom'].append(ate_odom)
        errors_per_type[seq_type]['pogo'].append(ate_pogo)
        errors_per_type[seq_type]['seq_id'].append(seq_id)


        # Compute the transform between the first and last poses
        odom_end_transform = np.linalg.inv(odom_poses[0]) @ odom_poses[-1]
        pogo_end_transform = np.linalg.inv(pogo_poses[0]) @ pogo_poses[-1]
        gt_end_transform = np.linalg.inv(gt_poses_interp[0]) @ gt_poses_interp[-1]

        odom_epe = np.linalg.norm((np.linalg.inv(gt_end_transform) @ odom_end_transform)[:2, 3])
        pogo_epe = np.linalg.norm((np.linalg.inv(gt_end_transform) @ pogo_end_transform)[:2, 3])
        epe_errors_per_type[seq_type]['odom'].append(odom_epe)
        epe_errors_per_type[seq_type]['pogo'].append(pogo_epe)
        epe_errors_per_type[seq_type]['seq_id'].append(seq_id)

        # Write the ATE and EPE to a text file
        df = pd.DataFrame.from_dict({'ATE': ate_pogo, 'EPE': pogo_epe}, orient='index').T
        df.to_csv(os.path.join("output", seq_id, seq_id + "_errors.csv"), index=False)


        # Display the results trajectories
        plt.figure(figsize=(6,6))
        plt.plot(odom_poses[:,0,3], odom_poses[:,1,3], label=kVizParams['dro']['label'], color=kVizParams['dro']['color'], linewidth=0.5)
        plt.plot(pogo_poses[:,0,3], pogo_poses[:,1,3], label=kVizParams['pogo']['label'], color=kVizParams['pogo']['color'], linewidth=0.5)
        plt.plot(gt_poses_interp[:,0,3], gt_poses_interp[:,1,3], label=kVizParams['gt']['label'], color=kVizParams['gt']['color'], linewidth=0.5)
        if loops is not None:
            for loop in loops.itertuples():
                time_i = utils.nameToTime(loop.scan_i_name)
                time_j = utils.nameToTime(loop.scan_j_name)
                id_i = np.argmin(np.abs(odom_times - time_i))
                id_j = np.argmin(np.abs(odom_times - time_j))
                # If that's the first loop in the sequence, add the label
                if loop == loops.itertuples().__iter__().__next__():
                    plt.plot([odom_poses[id_i, 0, 3], odom_poses[id_j, 0, 3]], [odom_poses[id_i, 1, 3], odom_poses[id_j, 1, 3]], alpha=0.5, label=kVizParams['loop']['label'], linewidth=0.5, color=kVizParams['loop']['color'])
                else:
                    plt.plot([odom_poses[id_i, 0, 3], odom_poses[id_j, 0, 3]], [odom_poses[id_i, 1, 3], odom_poses[id_j, 1, 3]], alpha=0.5, linewidth=0.5, color=kVizParams['loop']['color'])

        plt.legend(loc='upper left')
        plt.xlabel("X (m)")
        plt.ylabel("Y (m)")
        plt.axis('equal')
        plt.title(f"Trajectories for sequence {seq_id}")
        plt.savefig(os.path.join("output", seq_id, seq_id + "_trajectories.pdf"))
        #plt.show()
        plt.close()


    # Display all the errors avereage per type followed by per-sequence error
    for seq_type, errors in errors_per_type.items():
        print("\n\n\n\n==================\nSequence Type:", seq_type)
        odom_avg_ate = np.round(np.mean(errors['odom']), 2)
        pogo_avg_ate = np.round(np.mean(errors['pogo']), 2)
        odom_avg_epe = np.round(np.mean(epe_errors_per_type[seq_type]['odom']), 2)
        pogo_avg_epe = np.round(np.mean(epe_errors_per_type[seq_type]['pogo']), 2)
        print("    Average 2D Absolute Trajectory Error, RMSE ATE: odom", odom_avg_ate, "m, \tpogo:", pogo_avg_ate, "m")
        print("    Average 2D Endpoint Error, EPE: odom", odom_avg_epe, "m, \tpogo:", pogo_avg_epe, "m")
        for i in range(len(errors['seq_id'])):
            print("      Seq ID", errors['seq_id'][i], "ATE : odom", np.round(errors['odom'][i], 2), "m, \tpogo:", np.round(errors['pogo'][i], 2), "m, \tEPE: odom", np.round(epe_errors_per_type[seq_type]['odom'][i], 2), "m, \tpogo:", np.round(epe_errors_per_type[seq_type]['pogo'][i], 2), "m")



if __name__ == "__main__":
    main()