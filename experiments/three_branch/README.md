# NR-PANNs-MR (3 branches) — FIXED Jupyter package

## Đã sửa
- Dùng tên source riêng `mr_train.py`, `mr_model.py`, `mr_encoder.py` để tránh import nhầm `train.py` / `nr_panns.py` của 2 nhánh trong Jupyter.
- Xác nhận class `NRPANNsMR` và chữ ký `forward_features(waveform, return_reliability=False)` trước khi chạy.
- `DataLoader` sử dụng `pin_memory=False` và `num_workers=0` mặc định, tránh lỗi pin-memory CUDA ở môi trường người dùng.
- Cell kiểm tra CUDA tính toán ma trận thực thay vì chỉ kiểm tra `torch.cuda.is_available()`.
- Cell test feature map chỉ forward một lần, kiểm tra `[B,128,T,F]`, logits `[B,50]`, 3 reliability scores.
- Đường dẫn output riêng: `runs/nrpanns_mr_fixed` để không ghi đè experiment 2 nhánh.
- Train chỉ sau khi kiểm tra được GPU; `train_fold` khởi tạo lại từ pretrained AudioSet.

## Cách dùng
1. Giải nén toàn bộ vào **một thư mục mới** riêng, ví dụ `/home/robinson/robotic/PANNS/NRPANNs_MR_FIXED`.
2. Trong terminal: `cd /home/robinson/robotic/PANNS/NRPANNs_MR_FIXED && jupyter lab` rồi mở `NR_PANNs_MR_3branch_FIXED.ipynb`.
3. Trong notebook, sửa `ROOT` trỏ đến thư mục PANNs chứa `audioset_tagging_cnn/pytorch/models.py`, `checkpoints/Wavegram_Logmel_Cnn14_mAP=0.439.pth`, `data/ESC-50/`.
4. Restart Kernel, Run All hoặc chạy từng cell từ đầu. Nếu notebook mở từ nơi khác, sửa `SOURCE_DIR` ở cell import.
5. GPU lỗi `cudaErrorDevicesUnavailable`: kiểm tra `nvidia-smi`, dừng **chỉ** tiến trình cũ của bạn và restart kernel; việc tắt `pin_memory` không thể sửa GPU unavailable khi `.to('cuda')`.

## Giới hạn kiểm tra
Đã kiểm tra compile các Python modules/notebook và forward+backward trên encoder giả lập CPU, bao gồm gradient qua các module. Chưa chạy với checkpoint AudioSet / dữ liệu ESC-50 thật; chưa xác minh hiệu quả accuracy. Nhánh NRE-V4 chỉ được học gián tiếp từ classification loss; cần đánh giá noise robustness và ablation riêng.

## CLI tùy chọn
`python mr_train.py --root /home/robinson/robotic/PANNS --fold 5 --epochs 100 --batch-size 2 --workers 0 --no-amp`
