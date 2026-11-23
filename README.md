# AI analysis service

FastAPI service nội bộ cho NestJS. Service nhận metadata ảnh từ NestJS, tự đọc ảnh từ S3, chạy `forWeb.onnx` đa crop và callback kết quả về NestJS. Client và frontend không gọi service này.

Trên Render, triển khai bằng `ai-service/render.yaml` dưới dạng **Private Service**. Nó không có URL công khai; NestJS gọi bằng `AI_SERVICE_URL` trỏ tới Internal URL do Render cấp. Cả hai service phải cùng workspace và region.

Mỗi chiều dùng một khóa riêng:

- NestJS → AI: `AI_SERVICE_KEY`, gửi trong `X-AI-Key`.
- AI → NestJS: `AI_CALLBACK_SECRET`, gửi trong `X-AI-Key` đến `AI_CALLBACK_URL` cố định.

`AI_CALLBACK_URL` nên dùng Internal URL của NestJS, ví dụ `http://chicken-api-xxxx:10000/internal/ai/result`. Không truyền callback URL theo từng job, loại bỏ khả năng FastAPI bị yêu cầu gọi đến một địa chỉ tùy ý.

`forWeb.onnx` và `forWeb_meta.json` là model và metadata duy nhất cho serving. Không dùng checkpoint `.pth`, Hugging Face hoặc PyTorch. Mỗi ảnh chạy đúng sáu crop vuông trong một ONNX batch: toàn cảnh, bốn góc và trung tâm 80%. Model nhận JPEG/JPG và PNG sau khi NestJS kiểm tra magic bytes; WebP cũng được hỗ trợ.

## Run locally

Create `.env` from `.env.example` and configure S3 plus service/callback keys. Then run:

```bash
cd ai-service
source venv/bin/activate
pip install -r requirements.txt
uvicorn main:app --host 0.0.0.0 --port 8000
```

Health check: `GET /health`.

## Render

1. Tạo Blueprint từ `ai-service/render.yaml`, rồi nhập các biến có `sync: false` trong Render Dashboard.
2. Sao chép Internal URL của AI service vào biến NestJS `AI_SERVICE_URL` và thêm `/` nếu cần. Ví dụ: `http://chicken-ai-xxxx:10000`.
3. Đặt `AI_CALLBACK_URL` thành Internal URL của NestJS cộng `/internal/ai/result`.
4. Đặt cùng giá trị `AI_SERVICE_KEY` ở NestJS và AI; đặt cùng `AI_CALLBACK_SECRET` ở NestJS và AI, nhưng hai khóa phải khác nhau.

Blueprint mặc định dùng 2 CPU / 4 GB RAM và ONNX Runtime với hai CPU threads để giữ độ trễ thấp. File ONNX được đưa vào Docker image.
