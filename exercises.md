# K4 — Ngày 1: Bài Tập & Phản Ánh
## Khám Phá LLM API | Phiếu Thực Hành

**Thời lượng:** 14h00–18h00
**Cách làm:** Trả lời từng câu ngay sau khi hoàn thành block tương ứng —
đừng để dồn hết về cuối buổi. Thay dòng `*Câu trả lời của bạn*` bằng câu
trả lời thật (chấm tự động sẽ đếm số câu đã trả lời).

---

## Block 1 — API Cơ Bản (trả lời sau Checkpoint 1)

### Câu 1.1 — Độ nhạy của temperature
Gọi `call_openai` với temperature 0.0, 0.7, 1.2 và 1.8 dùng prompt
**"Hãy kể cho tôi một sự thật thú vị về Hà Nội."**

**Bạn nhận thấy quy luật gì qua bốn phản hồi? Ở mức nào phản hồi bắt đầu
kém mạch lạc?** (2–3 câu)
> Khi tăng temperature thì độ đa dạng trong response cũng sẽ tăng theo nhưng độ chính xác, mạch lạc giảm dần. Sự suy giảm mạch lạc xuất hiện khi temperature khoảng 1.2 - 1.5. 

### Câu 1.2 — Chọn temperature cho sản phẩm
**Bạn sẽ đặt temperature bao nhiêu cho trợ lý soạn thảo hợp đồng pháp lý,
và bao nhiêu cho trợ lý viết slogan quảng cáo? Giải thích khác biệt.**
> Khi setup parameter cho trợ lý soạn thảo hợp đồng pháp lý thì sẽ để temperature thấp nhất có thể để câu trả lời chính xác nhất (vì nếu sai sẽ ảnh hưởng pháp lý), còn khi viết slogan quảng cáo thì sẽ setup temperature một khoảng khá cao để đảm bảo sự đa dạng và thú vị cho slogan.

### Câu 1.3 — Đánh đổi chi phí
Kịch bản: 20.000 người dùng hoạt động mỗi ngày, mỗi người gọi API 2 lần,
mỗi lần trung bình ~500 token đầu ra.

**Ước tính chi phí mỗi ngày của model lớn so với model nhỏ cho workload này
(dựa trên bảng giá trong template). Nêu một trường hợp model lớn xứng đáng
với chi phí và một trường hợp model nhỏ là lựa chọn đúng:**
> 20.000 người dùng × 2 lần gọi/ngày × 500 token đầu ra ≈ 20 triệu token/ngày. Nếu dùng model lớn  thì chi phí khoảng $300/ngày, trong khi model nhỏ chỉ tốn khoảng $12/ngày — chênh lệch khoảng 25 lần. Model lớn xứng đáng khi tác vụ đòi hỏi suy luận phức tạp, độ chính xác cao (ví dụ tư vấn pháp lý, phân tích tài chính), còn model nhỏ là lựa chọn đúng cho các tác vụ đơn giản, lặp lại nhiều (ví dụ phân loại email, trả lời FAQ) vì tốc độ nhanh và chi phí thấp hơn nhiều mà chất lượng vẫn đủ dùng.

---

## Block 2 — System Prompt & Token (trả lời sau Checkpoint 2)

### Câu 2.1 — Sức mạnh của persona
Gọi `chat_with_system_prompt` hai lần với cùng câu hỏi
**"Giải thích máy học (machine learning) là gì?"** nhưng hai system prompt
khác nhau:
- "Bạn là một nhà thơ, trả lời mọi thứ bằng hình ảnh ví von, tránh thuật ngữ."
- "Bạn là kỹ sư phần mềm senior, trả lời chính xác, có ví dụ code khi phù hợp."

**Hai phản hồi khác nhau như thế nào (giọng văn, độ dài, mức kỹ thuật)?
Từ đó rút ra system prompt điều khiển được những khía cạnh nào của phản hồi?**
(3–4 câu)
> Với system prompt "nhà thơ", phản hồi thường dùng hình ảnh ẩn dụ, ít thuật ngữ kỹ thuật, đôi khi dài dòng vì phải xây dựng hình ảnh. Với system prompt "kỹ sư senior", phản hồi ngắn gọn, súc tích hơn, có định nghĩa chính xác, có thể kèm ví dụ code hoặc thuật ngữ như "supervised learning", "gradient descent". Qua đó có thể thấy system prompt điều khiển được giọng văn, mức độ kỹ thuật, cấu trúc trình bày và phần nào độ dài phản hồi — nhưng nội dung cốt lõi (khái niệm ML) vẫn được giữ đúng.

### Câu 2.2 — tiktoken vs đếm từ
Chọn một đoạn văn tiếng Việt ~150 từ. So sánh số token theo `count_tokens`
(tiktoken) với ước lượng `số từ / 0.75` mà Part 1 đã dùng.

**Hai con số chênh nhau bao nhiêu phần trăm? Nếu dùng ước lượng thô để dự
toán ngân sách API cho ứng dụng tiếng Việt, bạn sẽ dự toán thiếu hay thừa —
và vì sao?**
> Do tiktoken tách theo byte-pair encoding gốc cho tiếng Anh, mỗi từ tiếng Việt có dấu thường bị tách thành 2–4 token thay vì 1 token như tiếng Anh, nên số token thực tế theo tiktoken thường cao hơn đáng kể (thường 30–60%) so với ước lượng số từ / 0.75. Nếu dùng công thức ước lượng thô này để dự toán ngân sách, bạn sẽ dự toán thiếu — vì công thức đó vốn được hiệu chỉnh cho tiếng Anh, không phản ánh đúng đặc thù dấu thanh và cấu trúc âm tiết của tiếng Việt.

---

## Block 3 — Streaming & Độ Bền (trả lời sau Checkpoint 3)

### Câu 3.1 — Trải nghiệm người dùng với streaming
**Xét ba ứng dụng: (a) chatbot văn bản, (b) trợ lý giọng nói đọc to phản hồi,
(c) pipeline dịch tài liệu chạy ngầm ban đêm. Ứng dụng nào hưởng lợi nhiều
nhất từ streaming, ứng dụng nào không cần — và tại sao?** (1 đoạn văn)
> Trợ lý giọng nói (b) hưởng lợi nhiều nhất từ streaming vì có thể bắt đầu chuyển văn bản thành giọng nói ngay khi có vài từ đầu tiên, giảm độ trễ. Chatbot văn bản (a) cũng hưởng lợi rõ rệt vì người dùng thấy phản hồi "đang được gõ" thay vì chờ đợi im lặng, tạo cảm giác tương tác tự nhiên hơn. Ngược lại, pipeline dịch tài liệu chạy ngầm ban đêm (c) không cần streaming vì không có người dùng theo dõi trực tiếp

### Câu 3.2 — Vì sao backoff theo cấp số nhân?
**Khi API quá tải và hàng nghìn client cùng retry, exponential backoff giúp
gì so với delay cố định? Tra cứu thêm: kỹ thuật "jitter" (thêm độ trễ ngẫu
nhiên) giải quyết vấn đề gì còn sót lại?**
> Nếu dùng delay cố định, hàng nghìn client sẽ đồng loạt retry lại đúng sau khoảng thời gian giống nhau, tạo ra một đợt "sóng" request mới làm server tiếp tục quá tải — vấn đề lặp lại theo chu kỳ. Exponential backoff giúp giãn dần khoảng thời gian chờ giữa các lần retry (ví dụ 1s, 2s, 4s, 8s...), làm giảm mật độ request dồn về server theo thời gian, cho hệ thống có cơ hội hồi phục. Tuy nhiên nếu tất cả client đều tính toán delay giống hệt nhau, chúng vẫn có thể retry đồng bộ ở từng bước — đây là vấn đề còn sót lại mà "jitter" giải quyết: bằng cách thêm một khoảng ngẫu nhiên nhỏ vào mỗi lần delay, jitter làm phân tán thời điểm retry của các client, tránh hiện tượng "thundering herd" (đàn client cùng gõ cửa server một lúc).

---

## Block 4 — Mini-Project (trả lời sau Checkpoint 4)

### Câu 4.1 — Thiết kế persona
**Viết lại system prompt bạn dùng cho trợ lý của mình. Chỉ ra 2 chỗ trong
prompt mà nếu xóa đi, hành vi trợ lý sẽ thay đổi rõ rệt — và mô tả thay đổi
đó:**
> System prompt của tôi: "Bạn là trợ lý học tập thân thiện, luôn trả lời bằng tiếng Việt, giải thích ngắn gọn trước rồi mới đi vào chi tiết, và luôn hỏi lại nếu câu hỏi chưa rõ ràng."

Nếu xóa "luôn trả lời bằng tiếng Việt": trợ lý có thể chuyển sang trả lời bằng tiếng Anh khi người dùng hỏi bằng thuật ngữ tiếng Anh, gây khó hiểu cho người dùng phổ thông.
Nếu xóa "luôn hỏi lại nếu câu hỏi chưa rõ ràng": trợ lý sẽ tự suy đoán ý người dùng và trả lời ngay cả khi câu hỏi mơ hồ, dễ dẫn đến trả lời sai ý hoặc lạc đề.

### Câu 4.2 — Hạn chế & cải thiện
**Trợ lý của bạn giữ history 4 lượt cuối. Hãy mô tả một tình huống hội thoại
cụ thể mà giới hạn này khiến trợ lý trả lời sai/mất ngữ cảnh, và đề xuất một
cách khắc phục (ví dụ: tóm tắt các lượt cũ, tăng giới hạn có chọn lọc...):**
> Người dùng hỏi ở lượt 1 "Tôi đang lên kế hoạch du lịch Đà Lạt 3 ngày", sau đó trò chuyện thêm 5 lượt về chủ đề khác (thời tiết, ẩm thực...), đến lượt thứ 7 hỏi "Vậy tôi nên mang theo bao nhiêu bộ quần áo?" — vì history chỉ giữ 4 lượt gần nhất, trợ lý đã "quên" thông tin về chuyến đi Đà Lạt 3 ngày, dẫn đến trả lời chung chung hoặc hỏi lại thông tin đã cung cấp. Cách khắc phục: định kỳ tóm tắt các lượt hội thoại cũ thành một đoạn ngữ cảnh ngắn (summary) và luôn đính kèm summary này ở đầu prompt, thay vì xóa hẳn — vừa giữ được thông tin quan trọng vừa không làm prompt quá dài.

---

## Danh Sách Kiểm Tra Nộp Bài

- [ ] `python grade.py` — xem điểm tự động, mục tiêu ≥ 75/100
- [ ] Cả 4 checkpoint pytest đều pass
- [ ] Tất cả 9 câu trong file này đã được trả lời
- [ ] Đã copy bài làm vào folder `solution/`, push lên GitHub cá nhân và nộp link repo vào vlearn (theo hướng dẫn README)
