# Train
# 0 - 20000             done
# 20000 - 40000         done
# 40000 - 60000         done
# 60000 - 80000         done
# 80000 - 100000        done
# 100000 - 120000       done
# 120000 - 140000       done
# 140000 - 160000       done
# 160000 - 180000       done
# 180000 - 200000       done
# 200000 - 220000       done
# 220000 - 240000       done
# 240000 - 260000       done
# 260000 - 280000       done
# 280000 - 300001       done


export CUDA_VISIBLE_DEVICES="0"

python generate_5_facial_landmarks.py \
        --location=/mnt/4TData/vuquang/3d-face-rec/Deep3DFaceRecon_pytorch/datasets/12-05-2024/val \
        # --start_index=298122 \
        # --end_index=300001