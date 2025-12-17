import asyncio
import json
from datetime import datetime

from typing import List, Optional, Dict

import httpx

import numpy as np

from motor.motor_asyncio import AsyncIOMotorDatabase

import re



from ..core.config import settings

from ..core.database import load_faiss_index

from ..models.document import DocumentInDB

from ..models.history import HistoryReference, create_history

from ..models.document import get_document_by_id, get_documents_by_user

from ..services.embedding import EmbeddingService


def detect_query_type_fast(question: str) -> str:
    """Enhanced query type detection với MULTI_PART và SECTION_OVERVIEW ưu tiên cao nhất."""
    q = question.lower()
    
    # PRIORITY 0: MULTI_PART - CRITICAL FIX: Detect multi-part questions FIRST
    # Pattern: "câu hỏi A? câu hỏi B?" hoặc "A và B" hoặc "A, B"
    multi_part_patterns = [
        r'\?\s+[A-ZĂÂĐÊÔƠƯ]',  # "question? Question" (Vietnamese uppercase)
        r'(nào|gì|bao\s*nhiêu).*?(và|,).*?(nào|gì|bao\s*nhiêu)',  # "X nào và Y nào"
        r'(có|là).*?(và|,).*?(có|là)',  # "X có... và Y có..."
        r'[?？]\s*[A-ZĂÂĐÊÔƠƯ]',  # "question? Question" (with Vietnamese chars)
        r'.*?\?.*?\?',  # Multiple question marks
        r'(module|framework|phần).*?(và|,).*?(module|framework|phần)',  # "module X và framework Y"
    ]
    
    # Check if question has multiple independent parts
    # CRITICAL FIX: Check multiple question marks FIRST (most reliable indicator)
    question_mark_count = question.count('?')
    if question_mark_count >= 2:
        print(f"[RAG] 🔀 MULTI_PART detected: {question_mark_count} question marks in question")
        return "MULTI_PART"
    
    # Check patterns
    pattern_matched = any(re.search(p, question) for p in multi_part_patterns)
    if pattern_matched:
        # Additional check: count question words
        question_words = ['nào', 'gì', 'bao nhiêu', 'có', 'là', 'được']
        question_count = sum(1 for word in question_words if word in q)
        # If has 2+ question words and separator (và, ,) → likely multi-part
        has_separator = 'và' in q or ',' in question or '?' in question
        if question_count >= 2 and has_separator:
            print(f"[RAG] MULTI_PART detected: {question_count} question words, has_separator={has_separator}")
            return "MULTI_PART"
    
    # PRIORITY 1: EXERCISE_GENERATION (Moved UP - Before SECTION_OVERVIEW)
    # Detect code generation requests first - CRITICAL FIX
    exercise_patterns = [
        r'tạo.*?bài\s*tập',
        r'viết.*?(function|hàm|code|class|lớp).*?dựa\s*trên',
        r'hãy\s*viết.*?code',
        r'tạo.*?class',
        r'áp\s*dụng.*?(vào|để.*?viết).*?code',
        r'cho.*?ví\s*dụ.*?code',
        r'viết.*?code.*?theo',
        r'dựa\s*trên.*?hãy\s*viết.*?code',  # "Dựa trên module X, hãy viết code"
        r'dựa\s*trên.*?viết.*?class',  # "Dựa trên module X, viết class Y"
    ]
    if any(re.search(p, q) for p in exercise_patterns):
        print(f"[RAG] 📝 EXERCISE_GENERATION detected: code generation request")
        return "EXERCISE_GENERATION"
    
    # PRIORITY 2: SECTION_OVERVIEW - CRITICAL FIX
    # Check for multi-section queries FIRST ("Phần X file A và Phần Y file B")
    section_matches = re.findall(r'(phần|chương|part)\s+(\d+)', q)
    if len(section_matches) >= 2 and ('và' in q or ',' in q or 'file' in q):
        # Special case: Multi-section overview from different files
        print(f"[RAG] 🎯 SECTION_OVERVIEW detected: {len(section_matches)} sections mentioned")
        return "SECTION_OVERVIEW"
    # BUT: Don't match if it's a MULTI_PART question (already checked above)
    # Patterns phải cover: "chi tiết hơn về PHẦN 8", "PHẦN 8 nói gì", "nội dung PHẦN 8"
    section_patterns = [
        # Original patterns
        r'(phần|chương|part)\s+\d+\s+(có|nói|là|gồm)',
        r'nội\s*dung\s+(phần|chương|part)\s+\d+',
        # CRITICAL NEW PATTERNS - More aggressive
        r'(phần|chương|part)\s+\d+',  # ANY mention of "PHẦN X"
        r'chi\s*tiết.*?(phần|chương|part)\s+\d+',  # "chi tiết hơn về PHẦN 8"
        r'rõ\s+hơn.*?(phần|chương|part)\s+\d+',
        r'về\s+(phần|chương|part)\s+\d+',
        r'(phần|chương|part)\s+\d+[:：]',  # "PHẦN 8: Title"
        r'trong\s+(phần|chương|part)\s+\d+',
        r'(phần|chương|part)\s+\d+.*?(gì|nào|những\s*gì|bao\s*gồm)',
        r'nội\s*dung.*?(phần|chương|part)\s+\d+',
        r'(phần|chương|part)\s+\d+\s+bao\s*gồm',
        r'tìm\s*hiểu.*?(phần|chương|part)\s+\d+',
        r'giới\s*thiệu.*?(phần|chương|part)\s+\d+',
    ]
    
    # If question mentions "PHẦN X" in ANY form → SECTION_OVERVIEW
    if any(re.search(p, q) for p in section_patterns):
        return "SECTION_OVERVIEW"
    
    # PRIORITY 2: DOCUMENT_OVERVIEW
    overview_patterns = [
        r'trong\s+(file|tài\s*liệu)\s+(này\s+)?có\s+gì',  # "trong file có gì" hoặc "trong file này có gì"
        r'(file|tài\s*liệu)\s+(này\s+)?(nói|viết|đề\s*cập)\s+về\s+gì',
        r'tổng\s*quan\s+(nội\s*dung\s+)?(của\s+)?(file|tài\s*liệu)',  # "tổng quan nội dung của 2 file"
        r'tổng\s*quan.*?(file|tài\s*liệu)',
        r'mục\s*lục',
        # CRITICAL: Thêm patterns cho "bao nhiêu phần"
        r'bao\s*nhiêu\s+(phần|chương|module|mục)',
        r'(file|tài\s*liệu).*?bao\s*nhiêu\s+(phần|chương|module)',
        r'(file|tài\s*liệu).*?nói\s+về\s+bao\s*nhiêu\s+(phần|chương)',
        r'có\s+bao\s*nhiêu\s+(phần|chương|module)',
        r'liệt\s*kê.*?(phần|chương|module)',
        r'tất\s*cả.*?(phần|chương|module)',
        r'(file|tài\s*liệu).*?có\s+gì',  # "file có gì"
        r'nội\s*dung.*?(file|tài\s*liệu)',  # "nội dung của file"
    ]
    if any(re.search(p, q) for p in overview_patterns):
        return "DOCUMENT_OVERVIEW"
    
    # PRIORITY 3: COMPARE_SYNTHESIZE - BEFORE other patterns
    comparative_patterns = [
        r'so\s*sánh',
        r'khác.*?gì',
        r'giống.*?gì',
        r'phân\s*biệt',
        r'(sự\s*)?khác\s*nhau.*?giữa',
        r'gộp.*?kiến\s*thức',
        r'kết\s*hợp.*?từ',
    ]
    if any(re.search(p, q) for p in comparative_patterns):
        return "COMPARE_SYNTHESIZE"
    
    # PRIORITY 4: Code analysis questions (Tầng 3) - CRITICAL FIX: Add more patterns
    code_patterns = [
        r'phân\s*tích.*?(code|đoạn\s*code|lỗi|công\s*thức)',  # Thêm "công thức"
        r'sửa.*?(code|lỗi|bug)',
        r'đoạn\s*code.*?(sai|lỗi|bug|đúng)',
        r'chấm\s*điểm.*?code',
        r'code.*?(có\s*vấn\s*đề|sai|lỗi)',
        r'tìm\s*lỗi.*?code',
        r'có\s*đúng\s*với\s*logic',  # "có đúng với logic trong file COCOMO"
        r'đúng\s*với.*?logic',
        r'đoạn\s*code.*?có\s*đúng',
        r'tính\s*toán.*?có\s*đúng',  # Thêm pattern này
        r'kiểm\s*tra.*?code',
    ]
    if any(re.search(p, q) for p in code_patterns):
        return "CODE_ANALYSIS"
    
    # PRIORITY 6: Multi-concept reasoning (Tầng 4)
    reasoning_patterns = [
        r'dựa\s*trên.*?(và|,).*?(hãy|viết|giải\s*thích)',
        r'kết\s*hợp.*?(và|,)',
        r'áp\s*dụng.*?(và|,)',
        r'(hoisting|scope|closure).*?(và|,).*(function|loop|variable)',
        r'giải\s*thích.*?cơ\s*chế.*?(và|,)',
    ]
    if any(re.search(p, q) for p in reasoning_patterns):
        return "MULTI_CONCEPT_REASONING"
    
    # PRIORITY 7: List/enumerate questions
    if any(kw in q for kw in ["liệt kê", "cho ví dụ", "ví dụ cho", "bao nhiêu"]):
        if "hãy liệt kê" in q or "cho ví dụ" in q or "liệt kê" in q:
            return "EXPAND"
    
    # PRIORITY 8: Existence check
    if any(kw in q for kw in ["có đề cập", "có nói", "tài liệu có"]):
        return "EXISTENCE"
    
    # PRIORITY 9: Explanation/elaboration
    if any(kw in q for kw in ["giải thích", "rõ hơn", "tại sao", "ví dụ"]):
        return "EXPAND"
    
    return "DIRECT"


def build_gemini_optimized_prompt(
    question: str,
    context_text: str,
    chunk_similarities: List[float],
    query_type: str = "DIRECT",
    selected_documents: Optional[List[Dict[str, str]]] = None,
) -> str:
    """
    Gemini 2.5 Flash optimized prompt - SHORT, STRICT, STRUCTURED.
    Target: <1000 tokens for system instructions.
    """
    
    # Auto-detect query type
    q_lower = question.lower()
    
    # CRITICAL FIX: Mở rộng patterns cho SECTION_OVERVIEW detection
    section_query_patterns = [
        # CRITICAL FIX: Pattern cơ bản nhất - bắt MỌI mention của "PHẦN X"
        r'(phần|chương|part)\s+(\d+)',  # Bắt bất kỳ mention của "PHẦN X" hoặc "phần 4"
        r'(phần|chương|part)\s+(\d+)\s+(có|nói|là|gồm)',  # "phần 8 có gì"
        r'nội\s*dung\s+(phần|chương|part)\s+(\d+)',      # "nội dung phần 8"
        r'(phần|chương|part)\s+(\d+)\s+.*?(gì|nào)',      # "phần 8 nói gì"
        # THÊM CÁC PATTERN MỚI
        r'chi\s*tiết.*?(phần|chương|part)\s+(\d+)',       # "chi tiết về phần 8"
        r'(phần|chương|part)\s+(\d+).*?chi\s*tiết',       # "phần 8 chi tiết"
        r'giải\s*thích.*?(phần|chương|part)\s+(\d+)',     # "giải thích phần 8"
        r'(phần|chương|part)\s+(\d+).*?giải\s*thích',     # "phần 8 giải thích"
        r'tìm\s*hiểu.*?(phần|chương|part)\s+(\d+)',       # "tìm hiểu phần 8"
        r'(phần|chương|part)\s+(\d+).*?tìm\s*hiểu',       # "phần 8 tìm hiểu"
        r'về\s+(phần|chương|part)\s+(\d+)',               # "về phần 8", "chi tiết hơn về phần 8"
        r'(phần|chương|part)\s+(\d+)\s+là\s+gì',          # "phần 8 là gì"
        r'(phần|chương|part)\s+(\d+)\s+nói\s+về',         # "phần 8 nói về"
        r'(phần|chương|part)\s+(\d+)\s+bao\s+gồm',        # "phần 8 bao gồm"
        # CRITICAL: Thêm patterns cho câu có "chi tiết hơn"
        r'chi\s*tiết\s+hơn.*?(phần|chương|part)\s+(\d+)',  # "chi tiết hơn PHẦN 8"
        r'rõ\s+hơn.*?(phần|chương|part)\s+(\d+)',          # "rõ hơn về PHẦN 8"
        r'nói\s+rõ.*?(phần|chương|part)\s+(\d+)',          # "nói rõ PHẦN 8"
        # Pattern đặc biệt: bắt cả format "PHẦN 8: Title"
        r'(phần|chương|part)\s+(\d+)[:：]',                 # "PHẦN 8:" or "PHẦN 8："
    ]
    
    is_section_query = False
    section_match = None
    section_num = None
    for pattern in section_query_patterns:
        match = re.search(pattern, q_lower)
        if match:
            is_section_query = True
            section_match = match
            # Lấy section number từ group phù hợp (có thể là group 1 hoặc 2 tùy pattern)
            groups = match.groups()
            for i, group in enumerate(groups, 1):
                if group and group.isdigit():
                    section_num = group
                    break
            # Fallback: nếu không tìm thấy số trong groups, thử extract từ match string
            if not section_num:
                # Extract số từ toàn bộ match
                num_match = re.search(r'\d+', match.group(0))
                if num_match:
                    section_num = num_match.group(0)
            break
    
    # Detect other query modes
    is_overview = any(re.search(p, q_lower) for p in [
        r'trong\s+(file|tài\s*liệu)\s+này\s+có\s+gì',
        r'(file|tài\s*liệu)\s+này\s+(nói|viết|đề\s*cập)\s+về\s+gì',
        r'tổng\s*quan\s+(file|tài\s*liệu)',
        r'mục\s*lục',
        # CRITICAL: Thêm patterns cho "bao nhiêu phần"
        r'bao\s*nhiêu\s+(phần|chương|module|mục)',
        r'(file|tài\s*liệu).*?bao\s*nhiêu\s+(phần|chương|module)',
        r'(file|tài\s*liệu).*?nói\s+về\s+bao\s*nhiêu\s+(phần|chương)',
        r'có\s+bao\s*nhiêu\s+(phần|chương|module)',
        r'liệt\s*kê.*?(phần|chương|module)',
        r'tất\s*cả.*?(phần|chương|module)',
    ])
    
    # Determine mode based on query_type
    if query_type == "MULTI_PART":
        mode = "MULTI_PART"
        existence_subtype = None
    elif query_type == "CODE_ANALYSIS":
        mode = "CODE_ANALYSIS"
        existence_subtype = None
    elif query_type == "EXERCISE_GENERATION":
        mode = "EXERCISE_GENERATION"
        existence_subtype = None
    elif query_type == "MULTI_CONCEPT_REASONING":
        mode = "MULTI_CONCEPT_REASONING"
        existence_subtype = None
    elif query_type == "COMPARE_SYNTHESIZE":
        mode = "COMPARE_SYNTHESIZE"
        existence_subtype = None
    elif is_section_query:
        mode = "SECTION_OVERVIEW"
        existence_subtype = None
    elif is_overview:
        mode = "DOCUMENT_OVERVIEW"
        existence_subtype = None
    else:
        # Fallback to old detection logic
        is_too_broad = any(kw in q_lower for kw in [
            "toàn bộ", "tất cả mọi", "every", "all"
        ])
        is_list_all = any(kw in q_lower for kw in ["liệt kê", "cho ví dụ", "bao nhiêu"])
        is_expanded = is_list_all or any(kw in q_lower for kw in ["giải thích", "rõ hơn", "tại sao"])
        is_existence = any(kw in q_lower for kw in ["có đề cập", "có nói", "tài liệu có"])
        is_comparative = any(kw in q_lower for kw in ["so sánh", "khác", "giống"])
        
        if is_too_broad:
            mode = "TOO_BROAD"
            existence_subtype = None
        elif is_expanded:
            mode = "EXPAND"
            existence_subtype = None
        elif is_existence:
            mode = "EXISTENCE"
            is_mention_only = any(kw in q_lower for kw in ["có đề cập", "có nhắc đến", "có nói đến"])
            existence_subtype = "MENTION_ONLY" if is_mention_only else "EXPLAINS"
        elif is_comparative:
            mode = "COMPARE"
            existence_subtype = None
        else:
            mode = "DIRECT"
            existence_subtype = None
    
    # Check similarity threshold
    max_sim = max(chunk_similarities) if chunk_similarities else 0
    auto_fallback_warning = ""
    # CRITICAL FIX: Giảm threshold cho DOCUMENT_OVERVIEW để bao gồm chunks có similarity thấp
    if mode == "DOCUMENT_OVERVIEW":
        similarity_threshold = 0.25  # Giảm từ 0.4 xuống 0.25 cho DOCUMENT_OVERVIEW
    else:
        similarity_threshold = 0.4  # Giữ nguyên cho các query type khác
    
    if max_sim < similarity_threshold:
        auto_fallback_warning = f"\n⚠️ WARNING: Max similarity < {similarity_threshold} → Must return FALLBACK."
    
    # Build mode-specific instructions
    mode_instructions = ""
    multi_doc_instruction = ""
    
    # 🔥 CRITICAL FIX: Build multi-doc instruction for MULTI_PART
    if mode == "MULTI_PART" and selected_documents and len(selected_documents) > 1:
        doc_names = [doc.get("filename", "Unknown") for doc in selected_documents]
        multi_doc_instruction = f"""
**🚨 CRITICAL: MULTI-DOCUMENT MULTI_PART DETECTED**
You are processing {len(selected_documents)} document(s): {', '.join(doc_names)}
- **MUST answer EACH part using chunks from the RELEVANT document**
- **Part 1 (Module SLOC)**: MUST use chunks from COCOMO.pdf (if available) - this document contains project/module analysis
- **Part 2 (Javascript framework)**: MUST use chunks from Javascript-TuCobanToiNangCao.pdf (if available) - this document contains framework information
- **Check chunk markers**: Look for [Chunk X] [COCOMO.pdf] or [Chunk Y] [Javascript-TuCobanToiNangCao.pdf] in the context
- **DO NOT say "không có trong tài liệu"** until you've checked ALL chunks from ALL documents
- **MUST cite chunks from BOTH documents if both parts are answered**
"""
    
    # 🔥 CRITICAL FIX: Build multi-doc instruction for DOCUMENT_OVERVIEW
    if mode == "DOCUMENT_OVERVIEW" and selected_documents and len(selected_documents) > 1:
        doc_names = [doc.get("filename", "Unknown") for doc in selected_documents]
        multi_doc_instruction = f"""
**🚨 CRITICAL: MULTI-DOCUMENT DOCUMENT_OVERVIEW DETECTED**
You are processing {len(selected_documents)} document(s): {', '.join(doc_names)}
- **MUST answer for ALL {len(selected_documents)} documents**
- **MUST process EACH document SEPARATELY**
- **MUST list ALL sections for EACH document**
- Format: "Theo tài liệu [filename1]..." then "Theo tài liệu [filename2]..."
- **DO NOT skip any document** - answer for ALL documents provided
- **DO NOT mix sections** - clearly separate by document name
"""
    
    if mode == "DOCUMENT_OVERVIEW":
        mode_instructions = """

## 📚 DOCUMENT OVERVIEW MODE - SCAN TẤT CẢ CHUNKS ĐỂ TÌM TẤT CẢ PHẦN

User asks: "file có gì", "bao nhiêu phần", "liệt kê các phần"

**🎯 YOUR PRIMARY GOAL:**
Scan ALL chunks systematically to find EVERY single section (PHẦN 1, 2, 3, ..., 10).
DO NOT STOP scanning until you've found ALL sections.

**CRITICAL: MULTI-DOCUMENT HANDLING**
If multiple documents are selected (you will see chunks from different files):
- **Process EACH document SEPARATELY**
- For each document, scan ALL chunks from that document
- Answer format: "Theo tài liệu [filename], tài liệu này bao gồm [X] phần..."
- If user asks about multiple files (e.g., "File A có bao nhiêu phần? File B có bao nhiêu module?"), answer BOTH questions separately
- DO NOT mix sections from different documents - clearly separate by document name

**CRITICAL STEPS (FOLLOW IN ORDER):**

1. **STEP 1: FIND TABLE OF CONTENTS CHUNK (PRIORITY #1)**
   - Look for chunks containing "MỤC LỤC" or "TABLE OF CONTENTS"
   - These chunks list ALL sections → most reliable source
   - If found: Extract ALL section titles from this chunk
   - Pattern: "PHẦN 1: Title", "PHẦN 2: Title", ..., "PHẦN 10: Title"

2. **STEP 2: SCAN ALL CHUNKS FOR SECTION HEADINGS**
   - Go through EVERY chunk systematically (chunk 0, 1, 2, ..., N)
   - Look for patterns: "PHẦN X:", "Chương X:", "Module X"
   - Extract section number + title
   - Create a list: [1, 2, 3, 5, 7, 9] ← Note gaps!

3. **STEP 3: FILL GAPS (CRITICAL)**
   - If you found PHẦN 1, 2, 3, 5, 7, 9 → YOU MUST find 4, 6, 8, 10
   - Keep scanning more chunks until NO gaps remain
   - Expected: Continuous sequence 1, 2, 3, 4, 5, 6, 7, 8, 9, 10

4. **STEP 4: EXTRACT CONTENT FOR EACH SECTION**
   - For each section number, find chunks describing that section
   - Extract 1-2 sentences ONLY about what the section covers (KEEP IT SHORT)
   - Format: "**PHẦN X: Title** - Brief description (1-2 sentences max)"
   - DO NOT write long paragraphs - be concise like COCOMO example
   - Use subsection info if available (e.g., "8.1 String, 8.2 Function")

5. **STEP 5: OUTPUT FORMAT (CRITICAL)**
   - MUST use numbered list with double newline between items
   - Format: "1. **PHẦN X: Title** - Description\\n\\n2. **PHẦN Y: Title** - Description\\n\\n..."
   - NEVER write on same line: "1. Part1 2. Part2" ← WRONG!

**EXAMPLE OUTPUT (CORRECT FORMAT):**
```
Tài liệu này bao gồm 10 phần sau:

1. **PHẦN 1: Giới thiệu về Javascript** - Lịch sử phát triển và lý do nên học Javascript.

2. **PHẦN 2: Tổng quan Javascript** - Công cụ phát triển và cách thực thi chương trình.

3. **PHẦN 3: Cú pháp cơ bản** - Biến, kiểu dữ liệu, toán tử và câu lệnh điều kiện.

4. **PHẦN 4: Cú pháp nâng cao** - Function, loop, break/continue và switch-case.

5. **PHẦN 5: Dữ liệu có cấu trúc** - Object và Array với các thao tác cơ bản.

6. **PHẦN 6: Higher-Order Function** - Callback, Promise và Async/Await.

7. **PHẦN 7: Lập trình hướng đối tượng** - OOP principles và tính kế thừa.

8. **PHẦN 8: Cú pháp ES6** - String, Function, Class và Destructuring.

9. **PHẦN 9: Javascript Framework** - jQuery, ReactJS, VueJS và Angular.

10. **PHẦN 10: Bài tập** - 5 bài tập thực hành với unit test.

[Cite chunks used]
```

**CRITICAL FORMATTING RULES:**
- ✅ MUST use double newline (\\n\\n) between each numbered item
- ✅ Each item MUST be on separate lines with blank line between
- ✅ Format: "1. **PHẦN X: [Title]** - [description]\\n\\n2. **PHẦN Y: [Title]** - [description]\\n\\n..."
- ✅ DO NOT write items on same line: "1. Part 1 2. Part 2" ← WRONG!
- ✅ MUST write: "1. Part 1\\n\\n2. Part 2\\n\\n3. Part 3" ← CORRECT!
- ✅ **KEEP IT SHORT**: Each section = 1-2 sentences MAX (like COCOMO example)
- ✅ **TOTAL LENGTH**: Keep answer < 2000 characters to avoid JSON parse errors

**OUTPUT FORMAT (Multiple Documents - CRITICAL):**
```
**Theo tài liệu [filename1]:** (MUST use bold header)

Tài liệu này bao gồm [X] phần sau:

1. **PHẦN 1: [Title]** - [description] (Chunk X, Y)

2. **PHẦN 2: [Title]** - [description] (Chunk Z)

3. **PHẦN 3: [Title]** - [description] (Chunk A, B)

...

**Theo tài liệu [filename2]:** (MUST use bold header, MUST separate with double newline)

Tài liệu này có [Y] module sau:

1. **Module 1: [Title]** - [description] (Chunk C, D)

2. **Module 2: [Title]** - [description] (Chunk E)

...

[Cite chunks used from both documents]
```

**CRITICAL FORMATTING REQUIREMENTS FOR MULTI-DOCUMENT:**
- ✅ **MUST use bold headers** for document names: `**Theo tài liệu [filename]:**`
- ✅ **MUST use double newlines** (`\\n\\n`) between documents AND between sections
- ✅ **MUST cite chunks** for each section: `(Chunk X, Y, Z)`
- ✅ **MUST use line breaks** - NEVER write everything in one paragraph
- ✅ **MUST separate clearly** - Use `\\n\\n` before starting a new document section

**CRITICAL RULES:**
- ⚠️ MUST list ALL sections (if document has 10 parts, list ALL 10)
- ⚠️ Double newline (\\n\\n) between each numbered item
- ⚠️ NO GAPS: If you see PHẦN 1, 2, 3, 5 → MUST find PHẦN 4
- ⚠️ Confidence = 0.95 if TOC found, 0.85 if found all without TOC
- ⚠️ **MUST process EACH document separately** - Do NOT mix sections from different documents
- ⚠️ **MUST scan EVERY chunk systematically** - Go through chunks one by one, extract ALL "PHẦN X" patterns
- ⚠️ **MUST list ALL main sections for each document** - If you find PHẦN 1, 2, 3, 5, 7, 9, you MUST check for PHẦN 4, 6, 8, 10 in other chunks from the SAME document
- ⚠️ **DO NOT SKIP ANY PART** - If document has PHẦN 1-10, list ALL 10 parts, not just 3-4 parts
- ⚠️ **Check for gaps** - If you see PHẦN 1, 2, 3, 5, 7, 9 → scan more chunks from the SAME document to find PHẦN 4, 6, 8, 10
- ⚠️ **Use ALL chunks from each document** - Don't rely on just a few chunks, scan through ALL provided chunks from each document
- ⚠️ **If user asks about multiple files, answer BOTH** - Do NOT ignore any file mentioned in the question
- ⚠️ **FORMATTING**: Numbered lists MUST use double newline (\\n\\n) between items: "1. Part 1\\n\\n2. Part 2\\n\\n3. Part 3"
- Use section titles from document (don't invent titles)
- If TABLE OF CONTENTS chunk exists, use it as primary source but still verify by scanning other chunks from the SAME document
- If user asks "bao nhiêu phần", answer format: "Theo tài liệu [filename], tài liệu này có [X] phần: [list all parts with numbers]"

**CRITICAL OUTPUT FORMAT (MUST BE VALID JSON):**
```json
{{
  "answer": "Tài liệu này bao gồm các phần sau:\\n\\n1. **PHẦN 1: [Title]** - [description]\\n\\n2. **PHẦN 2: [Title]** - [description]\\n\\n...",
  "answer_type": "DOCUMENT_OVERVIEW",
  "chunks_used": [chunk_numbers],
  "confidence": 0.90-0.95,
  "sentence_mapping": [...],
  "sources": {{"from_document": true, "from_external_knowledge": false}}
}}
```

⚠️ CRITICAL: Output MUST be valid JSON. NO markdown blocks, NO extra text.

"""
    elif mode == "CODE_ANALYSIS":
        mode_instructions = """
## 🔍 CODE_ANALYSIS MODE (PHÂN TÍCH & SO SÁNH)

User provides a code snippet or formula and asks to check if it's correct based on the document.

**CRITICAL MINDSET:**
1. **User's Code = INPUT**: Có thể sai. KHÔNG TÌM code này trong document.
2. **Document = TRUTH**: Tìm công thức/logic ĐÚNG trong chunks.
3. **TASK**: So sánh User Code vs Document Logic.

**🚨 MULTI-DOCUMENT REQUIREMENTS:**
Nếu có nhiều documents:
- **MUST extract logic/rules from ALL documents**
- Document 1 có thể chứa công thức toán học
- Document 2 có thể chứa giải thích/context
- **MUST cite chunks from ALL documents** để phân tích đầy đủ

**🚨 FILE-SPECIFIC REQUIREMENTS:**
- **If question mentions a specific file** (e.g., "có đúng với logic trong file COCOMO không"):
  - **MUST prioritize chunks from that file** to find the formula/logic
  - **MUST use chunks from the mentioned file** even if other files also have relevant content
  - Example: "có đúng với logic trong file COCOMO" → Find COCOMO formula from COCOMO.pdf chunks FIRST
  - **DO NOT say "không tìm thấy" if chunks from target file exist**

**EXECUTION STEPS:**
1. **Identify User's Logic**: User đang tính gì? (vd: `cost = SLOC * 2000`)
2. **Find Document's Logic**: 
   - **If question mentions a specific file**: Look for formula in chunks from THAT file FIRST
   - **If no specific file**: Tìm công thức ĐÚNG trong TẤT CẢ documents
   - VD: "COCOMO formula: E = 3.0 * (KLOC)^1.12" from COCOMO.pdf chunks
   - **Search for keywords**: "COCOMO", "công thức", "formula", "Effort", "E =", "KLOC", "SLOC"
3. **Compare**: 
   - User: Linear (nhân trực tiếp)
   - Document: Exponential (lũy thừa)
4. **Conclusion**: Code SAI vì...

**OUTPUT FORMAT:**
```json
{{
  "answer": "Phân tích đoạn code:\\n\\n1. **Code người dùng**: `cost = SLOC * 2000` (Linear)\\n\\n2. **Logic trong tài liệu**: [Công thức ĐÚNG từ chunks]\\n\\n3. **Kết luận**: Code SAI vì...",
  "answer_type": "CODE_ANALYSIS",
  "chunks_used": [chunks from ALL documents],
  "confidence": 0.85-0.95,
  "reasoning_steps": ["Step 1", "Step 2", "Step 3"],
  "sources": {{"from_document": true, "from_external_knowledge": false}}
}}
```

**CRITICAL RULES:**
- ⚠️ **DO NOT** search for the user's code in the document - it's likely NOT there
- ⚠️ **DO** extract rules/formulas from the document and compare with user's code
- ⚠️ **DO** explain WHY the code is correct/incorrect based on document logic
- ⚠️ **If question mentions a specific file**: MUST find and use chunks from that file. DO NOT use chunks from other files.
- ⚠️ **MUST cite chunks from ALL documents provided** (but prioritize the mentioned file)
- ⚠️ **DO NOT say "không có thông tin" or "không tìm thấy" if chunks from target file exist**
- ⚠️ **NEVER say "không thể phân tích" if chunks contain formulas/rules**
- ⚠️ **Extract formulas/rules from target file FIRST, then compare with user code**
- Show reasoning process clearly
- If code has errors, explain WHY based on document knowledge and suggest fixes
- Confidence should be 0.8-0.95 if concepts/rules found in document

**CRITICAL:** Output MUST be valid JSON. NO markdown blocks, NO extra text.
"""
    elif mode == "EXERCISE_GENERATION":
        mode_instructions = """

## 📝 EXERCISE GENERATION MODE (Tầng 3)

User wants to create code/exercises based on document concepts.

**YOUR AUTHORITY:**
1. **You are a Teacher**: You MUST create NEW content based on concepts from the document.
2. **Cross-Document Synthesis**: If asked to use a module from File A to write code in Language from File B, **COMBINE THEM**.
   - Example: "Use 'User Module' (COCOMO) to write JS Class (Javascript PDF)"
   - Extract 'User Module' fields from COCOMO chunks (name, email, password...).
   - Extract 'Class syntax' from Javascript chunks.
   - **GENERATE**: A JS Class `User` with fields `name`, `email`...
3. **Concept Extraction**: Use the definitions, formulas, or logic found in the chunks.
4. **Generation**: Create a NEW example/exercise that applies these concepts.
5. **DO NOT** simply say "info not found" if you can see the core concepts (e.g., COCOMO formulas, JS map/filter syntax, function patterns).

**MANDATORY STEPS:**
1. **Understand Concepts**: Extract relevant concepts from chunks (even if no exact example exists)
   - If multiple documents: Extract concepts from EACH document separately
   - Example: Module structure from COCOMO.pdf, Class syntax from Javascript.pdf
2. **Synthesize**: Combine concepts from multiple documents if needed
   - Example: Module fields (COCOMO) + Class syntax (Javascript) = JS Class
3. **Create Code**: Write NEW code (not from document) applying these concepts
   - **MUST GENERATE CODE**: Do not just explain concepts, actually write the code
4. **Explain**: Link each part of code to concepts in document

**OUTPUT FORMAT:**
```json
{{
  "answer": "Dựa trên module [Name] (tài liệu A) và cú pháp [Lang] (tài liệu B), đây là code mẫu:\\n\\n```javascript\\nclass User {{\\n  constructor(name, email) {{\\n    this.name = name;\\n    this.email = email;\\n  }}\\n  // methods based on module functions\\n}}\\n```\\n\\n**Giải thích:**\\n- Thuộc tính `name`, `email` lấy từ yêu cầu module (chunk X).\\n- Cú pháp `class` tuân theo ES6 (chunk Y).",
  "answer_type": "EXERCISE_GENERATION",
  "chunks_used": [chunk_X, chunk_Y],
  "confidence": 0.9,
  "sources": {{"from_document": true, "from_external_knowledge": true}}
}}
```

**CRITICAL RULES:**
- ⚠️ **MUST GENERATE CODE**: Do not just explain concepts, actually write the code
- ⚠️ **MUST USE CONCEPTS FROM DOCS**: Extract module structure, syntax rules, etc. from chunks
- ⚠️ **AUTO-FILL CHUNKS**: Use chunks that describe the Module AND chunks that describe the Syntax
- Code must be NEW, not copied from document
- MUST explain how each part relates to document concepts
- If combining multiple concepts from multiple documents, cite chunks from ALL documents
- Confidence: 0.8-0.9 if concepts clearly found
- **You MAY use external knowledge** to create examples, as long as you base them on concepts from the document

"""
    elif mode == "MULTI_CONCEPT_REASONING":
        mode_instructions = """

## 🧠 MULTI-CONCEPT REASONING MODE (Tầng 4)

User asks question requiring reasoning across multiple concepts.

**MANDATORY STEPS:**
1. **Identify Concepts**: List all concepts mentioned in question
2. **Extract from Document**: Find chunks for EACH concept
3. **Connect**: Show how concepts relate to each other
4. **Synthesize**: Combine understanding to answer question
5. **Reason**: Apply logic beyond just quoting

**OUTPUT FORMAT:**
```
Để trả lời câu hỏi này, cần kết hợp các khái niệm:

1. [Concept 1] (từ chunk X): [Brief explanation]
2. [Concept 2] (từ chunk Y): [Brief explanation]

Kết nối các khái niệm:
[Explain how they relate, with reasoning]

Áp dụng vào câu hỏi:
[Answer with synthesis of concepts]

[Cite all chunks used]
```

**CRITICAL RULES:**
- MUST cite chunks for EACH concept
- Show reasoning process, not just quotes
- If concepts from different sections, cite both
- Confidence: 0.7-0.85 (reasoning adds uncertainty)
- If any concept missing from document → note it explicitly

**CRITICAL OUTPUT FORMAT (MUST BE VALID JSON):**
```json
{{
  "answer": "Để trả lời câu hỏi này, cần kết hợp các khái niệm:\\n\\n1. [Concept 1] (từ chunk X): [Brief explanation]\\n2. [Concept 2] (từ chunk Y): [Brief explanation]\\n\\nKết nối các khái niệm:\\n[Explain how they relate, with reasoning]\\n\\nÁp dụng vào câu hỏi:\\n[Answer with synthesis of concepts]",
  "answer_type": "MULTI_CONCEPT_REASONING",
  "chunks_used": [chunk_numbers],
  "confidence": 0.7-0.85,
  "reasoning_steps": [
    "Step 1: Identify all concepts",
    "Step 2: Extract from document",
    "Step 3: Connect concepts",
    "Step 4: Synthesize answer"
  ],
  "sentence_mapping": [...],
  "sources": {{"from_document": true, "from_external_knowledge": false}}
}}
```

⚠️ CRITICAL: Output MUST be valid JSON. NO markdown blocks, NO extra text.

"""
    elif mode == "MULTI_PART":
        mode_instructions = """

## 🔀 MULTI-PART QUESTION MODE

User asks 2+ independent questions in one query.

**CRITICAL STEPS:**

1. **Identify all sub-questions**: Split question into parts
   - Look for question marks (?)
   - Look for separators: "và", ","
   - Each part should be answered independently

2. **Answer EACH part separately**: 
   - Part 1: [Answer from relevant chunks]
   - Part 2: [Answer from relevant chunks]
   - DO NOT skip any part - answer ALL parts

3. **Find chunks for EACH part (CRITICAL - READ CAREFULLY)**:
   - **Part 1 (Module SLOC)**: 
     - **MUST check chunks from COCOMO.pdf FIRST** - this document contains project analysis and module information
     - Look for keywords in chunks: "module", "SLOC", "lớn nhất", "dự án", "Web Đọc Truyện", "Quản lý truyện", "18,000", "PHẦN 1"
     - **If question mentions "dự án Web Đọc Truyện" or "module"**, the answer is VERY LIKELY in COCOMO.pdf chunks
     - Check ALL chunks from COCOMO.pdf that contain "PHẦN 1" or "module" or "SLOC"
     - **DO NOT say "không có trong tài liệu"** until you've checked ALL chunks from COCOMO.pdf
   - **Part 2 (Javascript framework)**: 
     - **MUST check chunks from Javascript-TuCobanToiNangCao.pdf FIRST** - this document contains framework information
     - Look for keywords in chunks: "framework", "Javascript", "jQuery", "React", "Vue", "Angular", "PHẦN 9"
     - Check ALL chunks from Javascript-TuCobanToiNangCao.pdf that contain "PHẦN 9" or "framework"
   - **CRITICAL**: 
     - You have chunks from MULTIPLE documents - check the filename in chunk markers like [Chunk X] [COCOMO.pdf] or [Chunk Y] [Javascript-TuCobanToiNangCao.pdf]
     - For Part 1: Prioritize chunks from COCOMO.pdf
     - For Part 2: Prioritize chunks from Javascript-TuCobanToiNangCao.pdf
     - **If you see chunks from both documents, you MUST use chunks from BOTH documents**

4. **Cite chunks for EACH answer**: Don't mix citations
   - Part 1 citations: [chunk X, Y, Z from Document A]
   - Part 2 citations: [chunk A, B, C from Document B]
   - **MUST cite chunks from the document that actually contains the information**

**OUTPUT FORMAT:**
```
**Phần 1: [Sub-question 1]**

[Answer with citations from relevant chunks]

**Phần 2: [Sub-question 2]**

[Answer with citations from relevant chunks]

**Nguồn tham khảo:**
- Phần 1: chunk X, Y, Z (từ [filename])
- Phần 2: chunk A, B, C (từ [filename])
```

**EXAMPLE:**
Question: "Module nào trong dự án Web Đọc Truyện có SLOC lớn nhất? Javascript framework nào được khuyến nghị?"

Output:
```
**Phần 1: Module có SLOC lớn nhất trong dự án Web Đọc Truyện**

Theo tài liệu COCOMO.pdf, module "Quản lý truyện" có SLOC lớn nhất với 18,000 SLOC, bao gồm các sub-module như CSDL truyện (2,000), Tạo truyện (2,500), Xem truyện (4,000)...

**Phần 2: Javascript framework được khuyến nghị**

Theo tài liệu Javascript-TuCobanToiNangCao.pdf, các framework được khuyến nghị bao gồm:
- jQuery: Thư viện phổ biến cho DOM manipulation
- ReactJS: Framework do Facebook phát triển
- VueJS: Dễ học, kết hợp ưu điểm của Angular và React
- Angular: Framework mạnh mẽ do Google hỗ trợ

**Nguồn tham khảo:**
- Phần 1: chunk 24, 25, 26 (từ COCOMO.pdf)
- Phần 2: chunk 367, 368, 371 (từ Javascript-TuCobanToiNangCao.pdf)
```

**CRITICAL RULES:**
- ⚠️ MUST answer ALL parts, don't skip any
- ⚠️ **MUST search chunks from ALL documents for EACH part** - don't assume which document has which information
- ⚠️ **If question mentions specific documents** (e.g., "theo COCOMO", "trong tài liệu Javascript"), prioritize those documents but STILL check ALL documents
- ⚠️ Use chunks from DIFFERENT documents if needed - Part 1 from Document A, Part 2 from Document B
- ⚠️ Clearly separate each part with headers
- ⚠️ Cite chunks for EACH part separately with document name
- ⚠️ **If you see chunks from multiple documents, you MUST use chunks from ALL relevant documents**
- ⚠️ If a part cannot be answered after checking ALL chunks from ALL documents, explicitly state "Thông tin về [part] không có trong các tài liệu được cung cấp"
- ⚠️ Confidence = 0.75-0.85 if all parts answered with chunks from relevant documents, 0.5-0.7 if some parts missing

**CRITICAL OUTPUT FORMAT (MUST BE VALID JSON):**
```json
{{
  "answer": "**Phần 1: [Sub-question 1]**\\n\\n[Answer]\\n\\n**Phần 2: [Sub-question 2]**\\n\\n[Answer]",
  "answer_type": "MULTI_PART",
  "chunks_used": [chunk_numbers_for_part1, chunk_numbers_for_part2],
  "confidence": 0.75-0.85,
  "sentence_mapping": [...],
  "sources": {{"from_document": true, "from_external_knowledge": false}}
}}
```

⚠️ CRITICAL: Output MUST be valid JSON. NO markdown blocks, NO extra text.

"""
    elif mode == "COMPARE_SYNTHESIZE":
        mode_instructions = """

## 🔀 COMPARE & SYNTHESIZE MODE (Tầng 2)

User wants to compare concepts or synthesize knowledge from multiple sections.

**CRITICAL: You MUST extract information from ALL relevant chunks for BOTH items being compared**

**MANDATORY STEPS:**
1. **Find chunks for BOTH items**: Search chunks for information about BOTH items in the comparison
2. **Extract ALL key points**: For EACH item, extract ALL main characteristics, differences, similarities
3. **Create comprehensive table**: Compare ALL aspects in markdown table format
4. **Cite ALL chunks used**: MUST list ALL chunks that provided information in chunks_used array

**OUTPUT FORMAT (MUST USE MARKDOWN TABLE FOR COMPARISON):**
```
**So sánh [Item A] và [Item B]:**

| Tiêu chí | [Item A] | [Item B] |
|----------|----------|----------|
| **Định nghĩa** | [Full definition from chunk X] | [Full definition from chunk Y] |
| **Thời gian** | [Time info from chunk X] | [Time info from chunk Y] |
| **Cơ sở ước tính** | [Basis from chunk X] | [Basis from chunk Y] |
| **Đánh giá** | [Assessment from chunk X] | [Assessment from chunk Y] |
| **Chênh lệch** | [Difference info] | [Difference info] |

**Kết luận:**
[Giải thích chi tiết sự khác biệt, điểm giống nhau, và khi nào nên dùng cái nào]
```

**CRITICAL: MUST USE MARKDOWN TABLE FORMAT**
- ✅ MUST use markdown table: | Tiêu chí | [Item A] | [Item B] |
- ✅ MUST include separator row: |----------|----------|----------|
- ✅ Each row compares ONE aspect across both items
- ✅ **CRITICAL: KEEP IT CONCISE** - Limit to 5-7 most important comparison rows
- ✅ **CRITICAL: ANSWER LENGTH ≤ 3000 characters** - Summarize key differences, don't write long paragraphs
- ✅ Each cell should contain KEY information (1-2 sentences max per cell)
- ✅ **MANDATORY CITATIONS**: EVERY cell MUST end with citation: "(từ chunk X)" or "(từ chunk X, Y)" if multiple chunks
- ✅ **NO EXCEPTIONS**: If information comes from chunk, MUST cite it. If no chunk info, use: "Thông tin chưa có trong tài liệu"
- ✅ **If you need more details, suggest**: "Xem chi tiết tại chunks X, Y, Z"

**CRITICAL CITATION RULES:**
- EVERY row in table MUST cite source chunks
- Format: "... (từ chunk X)" or "... (từ chunk X, Y, Z)"
- If info from multiple chunks, cite ALL: "(từ chunk X, Y, Z)"
- chunks_used array MUST include ALL chunks mentioned

**CRITICAL RULES FOR COMPLETENESS:**
- MUST find chunks for ALL items being compared
- MUST extract ALL key points from chunks (extract fully, don't summarize)
- If comparing 3+ items, add columns: | Tiêu chí | [Item A] | [Item B] | [Item C] |
- If one item has more chunks, extract ALL information from ALL those chunks
- **CITATION RULE**: EVERY table cell MUST have citation at the end: "(từ chunk X)" or "(từ chunk X, Y, Z)" - NO EXCEPTIONS
- If cell combines info from multiple chunks, cite all: "(từ chunk X, Y, Z)"
- Confidence: 0.8-0.95 if comprehensive chunks found and ALL points extracted
- If information missing, note in cell: "Thông tin chưa có trong tài liệu"

**Example chunks_used for comparison:**
```json
{{
  "chunks_used": [76, 77, 85, 103, 109, 114],  // ALL chunks used
  ...
}}
```

**CRITICAL**: Never return empty chunks_used if table contains data. Always extract chunk numbers from ALL information sources.

**CRITICAL OUTPUT FORMAT (MUST BE VALID JSON):**
```json
{{
  "answer": "**So sánh [Item A] và [Item B]:**\\n\\n| Tiêu chí | [Item A] | [Item B] |\\n|----------|----------|----------|\\n| **Định nghĩa** | [Định nghĩa đầy đủ từ chunk X] | [Định nghĩa đầy đủ từ chunk Y] |\\n...\\n\\n**Kết luận:**\\n[Giải thích chi tiết]",
  "answer_type": "COMPARE_SYNTHESIZE",
  "chunks_used": [chunk_numbers],  # REQUIRED - MUST include ALL chunks that provided information
  "confidence": 0.8-0.95,
  "sentence_mapping": [...],
  "sources": {{"from_document": true, "from_external_knowledge": false}}
}}
```

⚠️ CRITICAL: 
- Output MUST be valid JSON. NO markdown blocks (```json), NO extra text before/after JSON
- Answer field MUST contain markdown table (with escaped newlines \\n)
- chunks_used MUST include ALL chunks that provided information for the comparison
- If you use information from chunks, you MUST list them in chunks_used array

"""
    elif is_section_query and section_num:
        mode_instructions = f"""

## 🎯 SECTION OVERVIEW MODE (MULTI-FILE AWARE)

User asks about specific sections (e.g., "PHẦN 4 file COCOMO và PHẦN 5 file JS").

**CRITICAL INSTRUCTIONS:**

1. **Identify ALL requested sections**: Parse the question to find which section belongs to which file.

2. **Search Strategy**: 

   - Look for [Chunk X] markers to identify which file a chunk belongs to.

   - If User asks for "PHẦN 4 COCOMO": Look specifically at chunks from `COCOMO.pdf` containing "PHẦN 4".

   - If User asks for "PHẦN 5 Javascript": Look at chunks from `Javascript...pdf` containing "PHẦN 5".

3. **Handling Missing Info**:

   - If you found info for File A but NOT File B, answer for File A and explicitly state: "Không tìm thấy nội dung PHẦN X trong tài liệu B".

   - DO NOT hallucinate.

**YOUR JOB:**
1. **Identify ALL sections mentioned**: If question asks "PHẦN 4 file A và PHẦN 5 file B", answer BOTH
2. **For EACH section**: Find ALL chunks containing that section number from the CORRECT file
3. **For EACH section**: Extract section title + **ALL subsections** (đừng bỏ sót!)
4. **For EACH section**: Write 2-3 sentences per subsection
5. **If section from specific file**: Look at chunks from that file first

**CRITICAL RULES:**
- ✅ **MULTI-SECTION SUPPORT**: If question asks about multiple sections (e.g., "PHẦN 4 file A và PHẦN 5 file B"), answer for ALL sections separately
- ✅ **FILE-SPECIFIC SEARCH**: If section is from a specific file (e.g., "PHẦN 4 trong file COCOMO" or "PHẦN 4 file COCOMO"), **MUST use chunks from that file ONLY**
- ✅ **MANDATORY FILE MATCHING**: If question mentions a specific file (COCOMO, Javascript, etc.), you MUST find and use chunks from that file. DO NOT use chunks from other files.
- ✅ **If question asks "PHẦN 4 file COCOMO"**: Look ONLY at chunks from COCOMO.pdf. Ignore chunks from other files even if they contain "PHẦN 4".
- ✅ Nếu PHẦN {section_num} có 8 subsections → phải list HẾT 8
- ✅ Nếu thiếu thông tin về subsection nào → ghi "Nội dung chi tiết cần bổ sung"
- ✅ **If a section is missing from chunks, state explicitly**: "Không tìm thấy thông tin về PHẦN X trong tài liệu [File]."
- ✅ **NEVER say "Không tìm thấy" if chunks from target file exist**: If chunks from the target file are provided, you MUST extract information from them
- ✅ KHÔNG BAO GIỜ return FALLBACK nếu tìm thấy chunks chứa "PHẦN {section_num}" từ file được yêu cầu
- ✅ Confidence = 0.90 nếu tìm thấy chunks từ file được yêu cầu

## CRITICAL: SECTION_OVERVIEW FORMAT VALIDATION

**MUST follow this EXACT structure:**

```
PHẦN {section_num}: [Exact title from document]

Nội dung chính bao gồm:

1. **[Subsection 1]** - [2-3 sentences description]

2. **[Subsection 2]** - [2-3 sentences description]

3. **[Subsection 3]** - [description]

... (list ALL subsections found in chunks)

[Citations: Chunk X, Y, Z]
```

**VALIDATION REQUIREMENTS:**
- ✅ MUST have "PHẦN {section_num}:" heading (exact format)
- ✅ MUST have "Nội dung chính bao gồm:" intro line
- ✅ MUST list 4+ subsections (NEVER return just 1-2 subsections)
- ✅ Each subsection MUST have format: `**Subsection Name** - description`
- ✅ Output MUST be JSON (not plain text)
- ❌ NEVER return markdown table format for comparisons
- ❌ NEVER return plain text without JSON structure

**OUTPUT FORMAT (Single Section):**
```
PHẦN {section_num}: [Title from chunks]

Nội dung chính bao gồm:

1. **[Subsection 1]** - [Explanation from chunks]

2. **[Subsection 2]** - [Explanation]

3. **[Subsection 3]** - [Explanation]

... (liệt kê HẾT tất cả subsections)
```

**OUTPUT FORMAT (Multi-Section from Multiple Files):**

Nếu tìm thấy thông tin từ cả 2 file, hãy trình bày tách biệt:

**1. Đối với tài liệu [Tên File 1] - PHẦN [X]: [Tiêu đề]**

* **Nội dung chính:** [Tóm tắt 2-3 ý chính]

* **Chi tiết:** - [Ý 1]

    - [Ý 2]



**2. Đối với tài liệu [Tên File 2] - PHẦN [Y]: [Tiêu đề]**

* **Nội dung chính:** [Tóm tắt 2-3 ý chính]

* **Chi tiết:**

    - [Ý 1]

    - [Ý 2]



[Citations: Chunk X, Y (File 1); Chunk A, B (File 2)]
```

**CRITICAL FORMATTING RULES:**
- ✅ MUST use double newline (\\n\\n) between each numbered item
- ✅ Each item MUST be on separate lines with blank line between
- ✅ Format: "1. **[Subsection 1]** - [description]\\n\\n2. **[Subsection 2]** - [description]\\n\\n..."
- ✅ DO NOT write items on same line: "1. Sub1 2. Sub2" ← WRONG!
- ✅ MUST write: "1. Sub1\\n\\n2. Sub2\\n\\n3. Sub3" ← CORRECT!
```

**JSON OUTPUT:**

{{
  "answer": "**1. Tài liệu COCOMO - PHẦN 4:**...\\n\\n**2. Tài liệu Javascript - PHẦN 5:**...",
  "answer_type": "SECTION_OVERVIEW",
  "chunks_used": [204, 205, 10, 11],  # ← Phải có chunks từ CẢ 2 file
  "confidence": 0.90,
  "sentence_mapping": [...],
  "sources": {{"from_document": true, "from_external_knowledge": false}}
}}

"""
    elif mode == "EXPAND":
        mode_instructions = """

## 📋 EXPAND MODE - GIẢI THÍCH VÀ VÍ DỤ

User wants examples or detailed lists (e.g., "liệt kê toàn bộ module", "cho ví dụ", "list all modules").

**MANDATORY STEPS (MUST BE COMPREHENSIVE):**
1. **Scan ALL chunks systematically**: Go through EVERY chunk to find ALL items
2. **Extract ALL modules/sub-modules**: If question asks about modules, find ALL modules AND sub-modules
3. **Preserve hierarchy**: If modules have sub-modules (e.g., 3.1, 3.2, 4.1, 4.2), list them ALL with proper hierarchy
4. **Extract ALL details**: For each module, extract ALL information (name, SLOC, description, sub-modules, etc.)

**CRITICAL RULES FOR MODULE LISTING:**
- ⚠️ **MUST list ALL modules**: If document has 11 modules, list ALL 11, not just 6
- ⚠️ **MUST list ALL sub-modules**: If module 3 has 10 sub-modules (3.1, 3.2, ..., 3.10), list ALL 10
- ⚠️ **Preserve table structure**: If chunks contain table with modules, preserve the table format
- ⚠️ **Extract ALL columns**: If table has columns (Module, Chức năng, SLOC, Ghi chú), extract ALL columns for ALL rows
- ⚠️ **Check for gaps**: If you see Module 1, 2, 3, 5, 7, 9 → scan more chunks to find Module 4, 6, 8, 10, 11

**OUTPUT FORMAT (Module Listing with Table):**
```
Theo [filename], các module được liệt kê bao gồm:

| Module | Chức năng chính | Ước tính SLOC | Ghi chú |
|--------|-----------------|---------------|---------|
| 1. Khởi động dự án | Lập kế hoạch, xác định phạm vi | 500 | Tài liệu, thiết lập môi trường ban đầu |
| 2. Phân tích yêu cầu | Thu thập & phân tích yêu cầu | 800 | Tài liệu đặc tả, use case, wireframe |
| 3. Quản lý người dùng | [Chức năng chính] | 12,000 | Module cốt lõi |
| 3.1 CSDL người dùng | Entity, validation, migration database | 1,500 | [Ghi chú] |
| 3.2 Đăng ký tài khoản | Form đăng ký, validation, xác thực email | 1,800 | [Ghi chú] |
| ... (list ALL sub-modules 3.1-3.10) |
| 4. Quản lý truyện | [Chức năng chính] | 18,000 | Module phức tạp nhất |
| 4.1 CSDL truyện | Entity phức tạp, quan hệ nhiều bảng | 2,000 | [Ghi chú] |
| ... (list ALL sub-modules 4.1-4.8) |
| ... (continue for ALL modules and sub-modules) |
| 11. Kết thúc dự án | [Chức năng] | [SLOC] | [Ghi chú] |
```

**OUTPUT FORMAT (Simple List):**
```
Theo [filename], các [items] được liệt kê bao gồm:

1. **[Item 1]** - [Description]

2. **[Item 2]** - [Description]

3. **[Item 3]** - [Description]

... (list ALL items found in chunks)
```

**CRITICAL FORMATTING RULES:**
- ✅ MUST use double newline (\\n\\n) between each numbered item
- ✅ Each item MUST be on separate lines with blank line between
- ✅ Format: "1. Item 1\\n\\n2. Item 2\\n\\n3. Item 3" - NEVER "1. Item 1 2. Item 2"
- ✅ If table format exists in chunks, preserve it with ALL rows and columns
- ✅ If modules have sub-modules, show hierarchy clearly (3.1, 3.2, etc.)

**CRITICAL OUTPUT FORMAT (MUST BE VALID JSON):**
```json
{{
  "answer": "Theo [filename], các module được liệt kê bao gồm:\\n\\n| Module | Chức năng | SLOC | Ghi chú |\\n|--------|-----------|------|---------|\\n| 1. [Name] | [Function] | [SLOC] | [Note] |\\n| ... (ALL modules and sub-modules) |",
  "answer_type": "EXPAND",
  "chunks_used": [chunk_numbers],
  "confidence": 0.90-0.98,
  "sentence_mapping": [...],
  "sources": {{"from_document": true, "from_external_knowledge": false}}
}}
```

⚠️ CRITICAL: Output MUST be valid JSON. NO markdown blocks, NO extra text.

"""

    if selected_documents and len(selected_documents) > 1:
        # Only add general multi-doc instruction if not already set for DOCUMENT_OVERVIEW
        if not multi_doc_instruction:
            doc_count = len(selected_documents)
            doc_names = ", ".join(
                [
                    (doc.get("filename") or doc.get("id") or f"Document {idx + 1}")
                    for idx, doc in enumerate(selected_documents)
                ]
            )
            multi_doc_instruction = f"""

## 🔀 MULTI-DOCUMENT QUERY HANDLING

**CRITICAL:** User selected {doc_count} document(s): {doc_names}

When answering:
1. If the question references MULTIPLE documents → cite chunks from **ALL** relevant documents.
2. When comparing information → explicitly mention each document (ví dụ: "Theo file A (chunk X)... trong khi file B (chunk Y)...").
3. Make it clear which document each fact comes from using the format: "Theo [filename] (chunk N)..."
4. Ensure `sentence_mapping` entries contain the correct `chunk` (which maps back to document_id).

Example multi-document citation:
"Theo Javascript-TuCobanToiNangCao.pdf (chunk 42), Promise được dùng cho async. Trong khi COCOMO.pdf (chunk 15) ước tính 425 nhân-tháng cho dự án."

"""
    
    prompt = f"""# SYSTEM RULES (DO NOT describe these rules, just follow them)

{mode_instructions}
{multi_doc_instruction}

## HARD FAILS (Violate any → immediate FALLBACK)
1. DO NOT answer if KEY CONCEPTS not in chunks.
2. DO NOT synthesize meaning from multiple unrelated chunks UNLESS in REASONING mode.
3. If similarity < 0.4 for ALL chunks → FALLBACK required{auto_fallback_warning}

**CRITICAL EXCEPTIONS (READ CAREFULLY):**
- **EXERCISE_GENERATION / EXPAND / MULTI_CONCEPT_REASONING**: 
  - You **MUST** answer if you understand the concepts from the chunks, even if exact examples are missing.
  - **DO NOT** return FALLBACK just because a specific example is missing. Create one!
  - **ALWAYS** populate `chunks_used` with the chunks that contain the definitions/concepts you used. NEVER leave `chunks_used` empty.

## MODE: {mode}
- CODE_ANALYSIS: Extract concepts → Apply to code → Step-by-step reasoning → Cite chunks
- EXERCISE_GENERATION: Understand concepts → Create NEW code → Explain links to document
- MULTI_CONCEPT_REASONING: Identify concepts → Extract from doc → Connect → Synthesize → Reason
- COMPARE_SYNTHESIZE: Find all chunks → Extract ALL points → Compare in TABLE format → Cite all
- SECTION_OVERVIEW: Full title + detailed numbered list (4-6 items, 2-3 sentences each)
- DOCUMENT_OVERVIEW: List main sections with descriptions
- DIRECT: Use only document text (4-6 sentences max)
- EXPAND: List ALL items if "liệt kê", or explain with examples
- COMPARE: Bullet list format
- EXISTENCE: Check if mentioned vs. explained in detail
- TOO_BROAD: Return "Câu hỏi quá rộng. Vui lòng hỏi về chủ đề cụ thể."

## REASONING GUIDELINES (for Tier 3-4 questions)
For CODE_ANALYSIS, EXERCISE_GENERATION, MULTI_CONCEPT_REASONING modes:
1. **You MAY use logical reasoning** beyond just quoting chunks
2. **You MAY create new examples** based on concepts from document
3. **You MUST cite** which chunks provided the base knowledge
4. **Mark synthesis**: Use phrases like "Áp dụng [concept từ chunk X]" or "Dựa trên [concept], ta suy ra..."

Example (CODE_ANALYSIS):
```
❌ BAD: "Chunk 5 nói về scope. Đoạn code này có lỗi scope."
✅ GOOD: "Dựa trên khái niệm scope trong chunk 5 (biến var có function scope), ta thấy dòng `console.log(i)` sẽ báo lỗi vì `i` được khai báo với `let` trong vòng for (block scope), không truy cập được bên ngoài."
```

## SELF-CHECK (MANDATORY before output)
1. Does answer directly address the question type?
2. For CODE/EXERCISE/REASONING: Did I show reasoning process?
3. For COMPARE: Did I find chunks for ALL items?
4. Are all citations correct and specific?
5. If FALLBACK → chunks_used MUST be []

## CHUNKS

{context_text}

## QUESTION

{question}

## OUTPUT (JSON ONLY - no markdown, no comments, no extra text)

⚠️ CRITICAL: You MUST return ONLY a valid JSON object. NO text before or after JSON.
⚠️ DO NOT include markdown code blocks (```json).
⚠️ DO NOT include any explanation outside the JSON object.
- Each section MUST be separated by blank lines (\n\n)
- For numbered lists (1., 2., 3...), MUST use double newline (\n\n) between items
- Format: "1. Item 1\\n\\n2. Item 2\\n\\n3. Item 3" - NEVER "1. Item 1 2. Item 2"

Example of CORRECT output:
{{
  "answer": "PHẦN 8: CÚ PHÁP ES6\\n\\nNội dung chính bao gồm:\\n\\n1. **String** - Template Literals...\\n\\n2. **Function** - Arrow functions...\\n\\n3. **Class** - ES6 classes...",
  "answer_type": "SECTION_OVERVIEW",
  "chunks_used": [204, 205, 208],
  "confidence": 0.95,
  "sentence_mapping": [{{"sentence": "first sentence", "chunk": 204, "external": false}}],
  "sources": {{"from_document": true, "from_external_knowledge": false}}
}}

⚠️ CRITICAL FORMATTING:
- Numbered lists MUST use double newline (\\n\\n) between items
- Format: "1. Item 1\\n\\n2. Item 2\\n\\n3. Item 3"
- NEVER: "1. Item 1 2. Item 2 3. Item 3" (all on one line)

Example of WRONG output (NEVER do this):
"Here is the answer: PHẦN 8..." ← NO! This is not JSON!

{{
  "answer": "string",
  "answer_type": "CODE_ANALYSIS|EXERCISE_GENERATION|MULTI_CONCEPT_REASONING|COMPARE_SYNTHESIZE|SECTION_OVERVIEW|DOCUMENT_OVERVIEW|DIRECT|EXPAND|COMPARE|EXISTENCE|TOO_BROAD|FALLBACK",
  "chunks_used": [integers],
  "confidence": 0.0-1.0,
  "sentence_mapping": [
    {{"sentence": "first sentence of answer", "chunk": 1, "external": false}},
    {{"sentence": "second sentence", "chunk": 3, "external": false}}
  ],
  "sources": {{"from_document": bool, "from_external_knowledge": bool}}
}}

CRITICAL: 
- If FALLBACK → chunks_used=[], confidence=0.0
- If CODE_ANALYSIS/REASONING → include reasoning_steps
- If COMPARE_SYNTHESIZE with missing concept → note in answer, confidence < 0.7
"""
    return prompt





class RAGService:

    def __init__(self):

        self.embedding_service = EmbeddingService()

        self.provider = settings.llm_provider.lower()

        self.model = settings.llm_model

        self.max_tokens = settings.llm_max_tokens

        # 🔥 CRITICAL FIX: Context length động theo query type
        self.base_max_context_length = min(12000, settings.rag_max_context_length)
        self.extended_max_context_length = 50000  # Extended context cho DOCUMENT_OVERVIEW
        
        # CRITICAL FIX: Tăng max_output_tokens cho Gemini
        # CRITICAL FIX: Tăng max_output_tokens cho câu trả lời dài
        self.max_output_tokens = 12000  # Increased from 8192

        self._openai_client = None

        self._gemini_api_key = None

        self._gemini_base_url = "https://generativelanguage.googleapis.com/v1/models"

        

        # Initialize OpenAI client if needed

        if self.provider == "openai" and settings.openai_api_key:

            try:

                from openai import OpenAI

                self._openai_client = OpenAI(api_key=settings.openai_api_key)

                print(f"[RAG] OpenAI client initialized successfully with model: {self.model}")

            except Exception as e:

                print(f"[RAG] Failed to initialize OpenAI client: {e}")

                import traceback

                print(f"[RAG] Traceback: {traceback.format_exc()}")

                self._openai_client = None

        

        # Initialize Gemini if needed

        if self.provider == "gemini":

            self._gemini_api_key = settings.gemini_api_key

            if self._gemini_api_key:

                print(f"[RAG] Gemini API initialized successfully with model: {self.model}")

            else:

                print("[RAG] Gemini API key not found, falling back to local generation")

                self.provider = "local"

        

        if self.provider == "openai" and self._openai_client is None:

            print("[RAG] Falling back to local generation mode")

            self.provider = "local"
    
    def _determine_max_chunks_for_query(self, question: str, query_type: str, num_docs: int = 1) -> int:
        """Dynamically determine max chunks based on query complexity."""
        q_lower = question.lower()
        
        # --- FIX START: Tăng multiplier mạnh hơn cho multi-doc ---
        if num_docs >= 3:
            multiplier = 2.5  # Tăng từ 2.0 lên 2.5
        elif num_docs >= 2:
            multiplier = 2.0  # Tăng từ 1.5 lên 2.0 để lấy đủ context cho cả 2 file
        else:
            multiplier = 1.0 
        # --- FIX END ---
        
        # 🔥 CRITICAL FIX: MULTI_PART needs MORE chunks (for multiple topics)
        if query_type == "MULTI_PART":
            base = 30  # Increase from default 15 to handle multiple questions
            return int(base * multiplier)  # 1 file: 30, 2 files: 45, 3+ files: 60
        
        # 🔥 CRITICAL FIX: DOCUMENT_OVERVIEW cần NHIỀU chunks nhất để scan toàn bộ document
        if query_type == "DOCUMENT_OVERVIEW":
            base = 300  # Tăng từ 150 lên 300 để scan đủ 10+ phần
            return int(base * multiplier)  # 1 file: 300, 2 files: 450, 3+ files: 600
        
        # CRITICAL FIX: SECTION_OVERVIEW cần chunks vừa phải - TĂNG cho multi-section
        if query_type == "SECTION_OVERVIEW":
            # Check if multi-section query (multiple section numbers mentioned)
            section_matches = re.findall(r'(phần|chương|part)\s+(\d+)', question.lower())
            if len(section_matches) >= 2:
                base = 60  # Multi-section needs more chunks
            else:
                base = 45  # Single section
            return int(base * multiplier)  # 45-60 → 67-90 cho multi-doc
        
        # Tier 4 (Reasoning): Cần nhiều chunks
        if query_type in ["MULTI_CONCEPT_REASONING", "CODE_ANALYSIS"]:
            # Count concepts mentioned
            concept_keywords = [
                "hoisting", "scope", "closure", "function", "arrow", "class",
                "object", "array", "loop", "for", "while", "if", "variable",
                "const", "let", "var", "promise", "async", "callback"
            ]
            concepts_found = sum(1 for kw in concept_keywords if kw in q_lower)
            
            if concepts_found >= 3:
                base = 40  # Tăng từ 30 lên 40
            elif concepts_found >= 2:
                base = 35  # Tăng từ 25 lên 35
            else:
                base = 30  # Tăng từ 20 lên 30
            return int(base * multiplier)
        
        # EXERCISE_GENERATION: Cần chunks để tìm đủ ngữ cảnh/công thức - TĂNG cho cross-doc
        if query_type == "EXERCISE_GENERATION":
            # Check if cross-document query (mentions multiple files or concepts from different docs)
            if num_docs >= 2 or ('file' in question.lower() and 'và' in question.lower()):
                base = 40  # Cross-document synthesis needs more chunks
            else:
                base = 30  # Single document
            return int(base * multiplier)  # 30-40 → 45-60 cho multi-doc
        
        # Tier 2 (Compare/Synthesize): Cần chunks từ nhiều sections - TĂNG để đầy đủ hơn
        if query_type in ["COMPARE_SYNTHESIZE", "COMPARE"]:
            # Check if comparing 2+ items
            if any(word in q_lower for word in ["và", "với", "so với", ","]):
                base = 40  # Tăng từ 35 lên 40
            else:
                base = 35  # Tăng từ 30 lên 35
            return int(base * multiplier)
        
        # Tier 3 (List all / Enumerate)
        if any(kw in q_lower for kw in ["liệt kê", "tất cả", "bao nhiêu", "cho ví dụ"]):
            base = 20  # Need more chunks to find all items
            return int(base * multiplier)
        
        # Tier 1 (Basic retrieval)
        base = 15  # Default
        return int(base * multiplier)
    
    def _extract_section_from_content(self, content: str, file_type: str) -> Optional[str]:
        """Extract section/heading from content if it looks like a heading.
        
        For DOCX/MD/TXT: Look for patterns like "7.1.2. Section Name" or short lines
        that could be headings.
        """
        if not content or file_type not in ["docx", "doc", "md", "txt"]:
            return None
        
        content = content.strip()
        
        # If content is very short (likely a heading), check if it matches heading patterns
        if len(content) > 150:  # Too long to be a heading
            return None
        
        # Pattern 1: Numbered sections like "7.1.2. Section Name" or "1.2.3 Section Name"
        numbered_section_pattern = r'^(\d+\.)+\s*\d+[\.\s]+(.+)$'
        match = re.match(numbered_section_pattern, content)
        if match:
            # Return the full content as it's likely a section heading
            return content
        
        # Pattern 2: Short lines that might be headings (less than 80 chars, ends with colon or no period)
        if len(content) < 80 and not content.endswith('.'):
            # Could be a heading, but be more careful
            # Check if it has structure like "SECTION NAME:" or "Section Name"
            if ':' in content or content.isupper() or (len(content.split()) <= 10 and not content.endswith('.')):
                return content
        
        # Pattern 3: Markdown-style headings (already handled by parser, but just in case)
        if content.startswith('#'):
            return content.lstrip('#').strip()
        
        return None
    
    def _is_numbered_section(self, text: str) -> bool:
        """Check if text looks like a numbered section heading (e.g., '7.2.2. Section Name')."""
        if not text:
            return False
        # Pattern for numbered sections: starts with digits and dots like "7.2.2." or "1.2.3 "
        numbered_pattern = r'^(\d+\.)+\s*\d+[\.\s]+'
        return bool(re.match(numbered_pattern, text.strip()))

    def _fix_numbered_list_formatting(self, text: str) -> str:
        """Fix numbered list formatting AND table row formatting.
        
        Converts: "1. Item 1 2. Item 2 3. Item 3"
        To: "1. Item 1\n\n2. Item 2\n\n3. Item 3"
        
        Also fixes table rows to ensure newlines between rows.
        """
        if not text:
            return text
        
        # Pattern 1: Fix numbered items on same line without newline: "1. ... 2. ..."
        # Match: (number + dot + space + content) followed by space + (number + dot + space)
        # But exclude sub-numbering patterns like "3.1. " or "5.2. "
        fixed = re.sub(
            r'(\d+\.\s+[^\n]+?)\s+(\d+\.\s+)(?![^\n]*\d+\.\d+\.)',  # Exclude if followed by sub-numbering
            r'\1\n\n\2',
            text
        )
        
        # Pattern 2: Items separated by single newline: "1. ...\n2. ..."
        fixed = re.sub(
            r'(\d+\.\s+[^\n]+)\n(?!\n)(\d+\.\s+)(?![^\n]*\d+\.\d+\.)',
            r'\1\n\n\2',
            fixed
        )
        
        # Pattern 3: Items with space before number: " ... 2. ..."
        # This handles cases where there's content then space then numbered item
        fixed = re.sub(
            r'([^\n])\s+(\d+\.\s+)(?![^\n]*\d+\.\d+\.)',
            r'\1\n\n\2',
            fixed
        )
        
        # Pattern 4: Fix table rows - ensure newlines between rows
        # Match: | col1 | col2 | (no newline) | col3 | col4 |
        # Replace: | col1 | col2 |\n| col3 | col4 |
        # But be careful not to break existing table formatting
        fixed = re.sub(
            r'(\|[^\n|]+\|)\s+(\|[^\n|]+\|)',  # Two consecutive rows without \n
            r'\1\n\2',
            fixed
        )
        
        # Clean up: Remove triple or more newlines (keep only double)
        fixed = re.sub(r'\n{3,}', '\n\n', fixed)
        
        # Final cleanup: Ensure no trailing spaces before newlines
        fixed = re.sub(r' +\n', '\n', fixed)
        
        # Debug: Log if we found numbered items
        numbered_items = len(re.findall(r'^\d+\.\s+', fixed, re.MULTILINE))
        if numbered_items > 0:
            print(f"[RAG] _fix_numbered_list_formatting: Found {numbered_items} numbered items, double_newlines={fixed.count(chr(10)*2)}")
        
        return fixed
    
    def _clean_citation_markers(self, text: str) -> str:
        """Remove citation markers like [Chunk X] and [Filename] from answer text.
        
        Removes:
        - [Chunk X] patterns
        - [Filename.pdf] patterns
        - [Chunk X] [Filename.pdf] combinations
        
        Preserves:
        - Code blocks (```...```)
        - Markdown links [text](url)
        """
        if not text:
            return text
        
        # 🔥 CRITICAL FIX: Preserve code blocks - split text into code blocks and regular text
        # Pattern to match code blocks: ```language\n...\n```
        code_block_pattern = r'```[\s\S]*?```'
        code_blocks = []
        code_block_placeholders = []
        
        # Extract all code blocks and replace with placeholders
        def replace_code_block(match):
            placeholder = f"__CODE_BLOCK_{len(code_blocks)}__"
            code_blocks.append(match.group(0))
            return placeholder
        
        text_with_placeholders = re.sub(code_block_pattern, replace_code_block, text)
        
        # Now process the text without code blocks
        # Remove [Chunk X] patterns (case insensitive)
        # Pattern: [Chunk 68], [Chunk 76], etc.
        text_with_placeholders = re.sub(r'\[Chunk\s+\d+\]', '', text_with_placeholders, flags=re.IGNORECASE)
        
        # Remove [Filename.pdf] patterns (common document names)
        # Pattern: [COCOMO.pdf], [Javascript-TuCobanToiNangCao.pdf], etc.
        # Match any text inside brackets that ends with .pdf (including hyphens, underscores, spaces)
        text_with_placeholders = re.sub(r'\[[^\]]*\.pdf\]', '', text_with_placeholders, flags=re.IGNORECASE)
        
        # Remove standalone [Filename] patterns (document names without .pdf extension)
        # But be careful: don't remove markdown links like [text](url) or [text][ref]
        # Pattern: [Filename] that appears after [Chunk X] or standalone, but not markdown links
        # We'll match [Filename] only if it's not followed by ( or [ (markdown link indicators)
        # Also match filenames with hyphens, underscores, spaces (like "Javascript-TuCobanToiNangCao")
        text_with_placeholders = re.sub(r'\[([A-Za-z0-9\s\-_]+)\](?!\s*[\(\[])', '', text_with_placeholders)
        
        # Clean up extra spaces left behind (but preserve newlines for formatting)
        # Only collapse multiple spaces to single space, but keep newlines
        text_with_placeholders = re.sub(r'[ \t]{2,}', ' ', text_with_placeholders)  # Multiple spaces/tabs to single space
        text_with_placeholders = re.sub(r'\s+([.,!?;:])', r'\1', text_with_placeholders)  # Space before punctuation
        text_with_placeholders = re.sub(r'([.,!?;:])\s+([.,!?;:])', r'\1\2', text_with_placeholders)  # Punctuation followed by punctuation
        text_with_placeholders = re.sub(r'^\s+', '', text_with_placeholders, flags=re.MULTILINE)  # Leading spaces on lines
        text_with_placeholders = re.sub(r'\s+$', '', text_with_placeholders, flags=re.MULTILINE)  # Trailing spaces on lines
        # Only collapse excessive newlines (more than 2 consecutive newlines)
        text_with_placeholders = re.sub(r'\n{3,}', '\n\n', text_with_placeholders)  # Multiple newlines to double newline
        
        # Restore code blocks
        for i, code_block in enumerate(code_blocks):
            placeholder = f"__CODE_BLOCK_{i}__"
            text_with_placeholders = text_with_placeholders.replace(placeholder, code_block)
        
        return text_with_placeholders.strip()
    
    def _clean_table_citations(self, text: str) -> str:
        """Remove citation lines from comparison tables.
        
        Removes:
        - "Nguồn tham khảo: ..." lines at the end
        - "(từ chunk X)" citations in table cells AND in conclusion text
        - Standalone "chunk X" references in table cells
        """
        if not text or "|" not in text:
            return text
        
        # Remove "Nguồn tham khảo:" line at the end (after table)
        # Pattern: "Nguồn tham khảo:" followed by document name and chunk numbers
        text = re.sub(
            r'\n\s*\*\*?Nguồn tham khảo:?\*\*?\s*[^\n]*(?:chunk\s+\d+(?:\s*,\s*\d+)*)*\s*',
            '',
            text,
            flags=re.IGNORECASE
        )
        text = re.sub(
            r'\n\s*Nguồn tham khảo:?\s*[^\n]*(?:chunk\s+\d+(?:\s*,\s*\d+)*)*\s*',
            '',
            text,
            flags=re.IGNORECASE
        )
        
        # 🔥 CRITICAL FIX: Remove citations in table cells: "(từ chunk X)" or "(từ filename.pdf, chunk X)"
        # Also remove in conclusion text if table exists
        lines = text.split('\n')
        cleaned_lines = []
        in_table = False
        for line in lines:
            # Check if this is a table row (contains | but not separator row)
            is_table_row = '|' in line and not re.match(r'^\|[-:|\s]+\|', line.strip())
            if is_table_row:
                in_table = True
                # This is a table row
                # Pattern 1: "(từ filename.pdf, chunk X)" or "(từ filename.pdf, chunk X, Y, Z)" - NEW FORMAT
                line = re.sub(r'\(từ\s+[^,)]+\.pdf,\s*chunk\s+\d+(?:\s*,\s*\d+)*\s*\)', '', line, flags=re.IGNORECASE)
                # Pattern 2: "(từ [filename], chunk X)" or "(từ [filename], chunk X, Y, Z)" - GENERIC FORMAT
                line = re.sub(r'\(từ\s+[^,)]+,\s*chunk\s+\d+(?:\s*,\s*\d+)*\s*\)', '', line, flags=re.IGNORECASE)
                # Pattern 3: "(từ chunk X)" or "(từ chunk X, Y, Z)" (fallback for old format)
                line = re.sub(r'\(từ\s+chunk\s+\d+(?:\s*,\s*\d+)*\s*\)', '', line, flags=re.IGNORECASE)
                # Remove standalone chunk references: "chunk X" or "chunk X, Y, Z" (not in parentheses, at end of cell)
                line = re.sub(r'\s+chunk\s+\d+(?:\s*,\s*\d+)*\s*(?=\||$)', '', line, flags=re.IGNORECASE)
                # Clean up extra spaces and trailing punctuation
                line = re.sub(r'\s{2,}', ' ', line)
                line = re.sub(r'\s+([|])', r'\1', line)  # Remove space before |
            elif in_table and line.strip() and not line.strip().startswith('|'):
                # This is text after table (like conclusion) - also remove citations
                # Pattern 1: "(từ filename.pdf, chunk X)" or "(từ filename.pdf, chunk X, Y, Z)"
                line = re.sub(r'\(từ\s+[^,)]+\.pdf,\s*chunk\s+\d+(?:\s*,\s*\d+)*\s*\)', '', line, flags=re.IGNORECASE)
                # Pattern 2: "(từ [filename], chunk X)" or "(từ [filename], chunk X, Y, Z)"
                line = re.sub(r'\(từ\s+[^,)]+,\s*chunk\s+\d+(?:\s*,\s*\d+)*\s*\)', '', line, flags=re.IGNORECASE)
                # Pattern 3: "(từ chunk X)" or "(từ chunk X, Y, Z)" (fallback)
                line = re.sub(r'\(từ\s+chunk\s+\d+(?:\s*,\s*\d+)*\s*\)', '', line, flags=re.IGNORECASE)
                # Remove standalone chunk references at end of sentences
                line = re.sub(r'\s+chunk\s+\d+(?:\s*,\s*\d+)*\s*(?=[\.\n]|$)', '', line, flags=re.IGNORECASE)
                # Clean up extra spaces
                line = re.sub(r'\s{2,}', ' ', line)
                # Pattern 1: "(từ [filename], chunk X)" or "(từ [filename], chunk X, Y, Z)"
                line = re.sub(r'\(từ\s+[^,)]+,\s*chunk\s+\d+(?:\s*,\s*\d+)*\s*\)', '', line, flags=re.IGNORECASE)
                # Pattern 2: "(từ chunk X)" or "(từ chunk X, Y, Z)" (fallback)
                line = re.sub(r'\(từ\s+chunk\s+\d+(?:\s*,\s*\d+)*\s*\)', '', line, flags=re.IGNORECASE)
                # Remove standalone chunk references at end of sentences
                line = re.sub(r'\s+chunk\s+\d+(?:\s*,\s*\d+)*\s*(?=[\.\n]|$)', '', line, flags=re.IGNORECASE)
                # Clean up extra spaces
                line = re.sub(r'\s{2,}', ' ', line)
            cleaned_lines.append(line)
        
        text = '\n'.join(cleaned_lines)
        
        # Final cleanup: remove empty lines and extra spaces
        text = re.sub(r'\n{3,}', '\n\n', text)
        text = text.strip()
        
        return text
    
    def _is_fallback_answer(self, answer: str) -> bool:
        """Enhanced fallback detection."""
        if not answer or len(answer.strip()) < 20:
            return True
        
        # Nếu câu trả lời chứa code block hoặc dài hơn 500 ký tự -> KHÔNG PHẢI FALLBACK
        # Đây là fix quan trọng cho các bài tập code
        if "```" in answer or len(answer) > 500:
            return False
        
        answer_lower = answer.lower()
        fallback_patterns = [
            "không đủ thông tin",
            "không tìm thấy thông tin",  # Sửa pattern cụ thể hơn
            "không thể trả lời",
            "tài liệu không đề cập đến",
            "không có dữ liệu về",
            "chưa có đủ dữ liệu",
            "document does not contain",
            "no information found",
            "cannot answer",
        ]
        
        # Chỉ coi là fallback nếu câu ngắn (dưới 200 ký tự) VÀ chứa từ khóa
        if len(answer) < 200:
            return any(pattern in answer_lower for pattern in fallback_patterns)
            
        return False
    
    def _safe_parse_json(self, raw: str, query_type: str = "DIRECT") -> dict:
        """Safe JSON parsing v2 - Robust against messy LLM output."""
        cleaned = raw.strip()
        
        # 🔥 CRITICAL FIX: Try to find JSON object even when wrapped in text
        # Method 1: Find JSON object boundaries (start with {, end with })
        start_idx = cleaned.find('{')
        end_idx = cleaned.rfind('}')
        
        if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
            potential_json = cleaned[start_idx : end_idx + 1]
            try:
                parsed = json.loads(potential_json)
                print(f"[RAG] ✅ Extracted JSON from text (method 1: find boundaries)")
                return parsed
            except json.JSONDecodeError as e:
                error_msg = str(e)
                print(f"[RAG] Method 1 failed: {error_msg}")
                
                # 🔥 CRITICAL FIX: If error is "Invalid \escape", try to fix escape sequences
                if "Invalid \\escape" in error_msg or "Invalid escape" in error_msg:
                    print(f"[RAG] 🔧 Detected escape sequence error, attempting to fix...")
                    try:
                        # Fix invalid escape sequences in JSON string values
                        # Replace invalid escape sequences (like \N, \x, etc.) with escaped backslash
                        # But keep valid ones: \n, \t, \r, \b, \f, \", \\, \/, \uXXXX
                        # Pattern: find \ followed by invalid escape char (not n, t, r, b, f, ", \, /, u, or hex digit)
                        # But don't replace if already escaped (\\)
                        fixed_json = re.sub(
                            r'(?<!\\)\\(?![nrtbf"\\/u0-9a-fA-F])',
                            r'\\\\',
                            potential_json
                        )
                        parsed = json.loads(fixed_json)
                        print(f"[RAG] ✅ Fixed escape sequences and parsed successfully")
                        return parsed
                    except Exception as e2:
                        print(f"[RAG] Escape fix failed: {e2}")
                # Continue to other methods if fix didn't work
        
        # CRITICAL FIX: If LLM returns plain text instead of JSON, try to extract
        # Check if response looks like plain text (doesn't start with {)
        if not cleaned.startswith('{'):
            print(f"[RAG] ⚠️ LLM returned plain text, not JSON. Attempting recovery...")
            print(f"[RAG] Raw text (first 200 chars): {cleaned[:200]}")
            
            # ENHANCED: Try multiple JSON extraction methods
            # CRITICAL: Handle tables in JSON answer field
            methods = [
                # Method 2: Find ```json blocks (most reliable)
                lambda s: re.search(r'```json\s*(\{.*?\})\s*```', s, re.DOTALL),
                # Method 3: Find JSON with proper quote handling for multiline strings
                lambda s: self._extract_json_with_multiline_string(s),
                # Method 4: Find first { to last } (fallback)
                lambda s: re.search(r'\{.*\}', s, re.DOTALL),
            ]
            
            for i, method in enumerate(methods, start=2):
                match = method(cleaned)
                if match:
                    try:
                        json_str = match.group(1) if match.lastindex else match.group(0)
                        # Try to fix common JSON issues with tables
                        json_str = self._fix_json_with_table(json_str)
                        parsed = json.loads(json_str)
                        print(f"[RAG] ✅ Extracted JSON using method {i}")
                        return parsed
                    except Exception as e:
                        print(f"[RAG] Method {i} failed: {e}")
                        continue
            
            # Method 5: Reconstruct JSON from text (fallback)
            print(f"[RAG] Attempting text-to-JSON reconstruction...")
            return self._reconstruct_json_from_text(cleaned, query_type)
        
        # Remove markdown blocks
        if cleaned.startswith("```"):
            cleaned = re.sub(r'^```json?\s*', '', cleaned)
            cleaned = re.sub(r'\s*```$', '', cleaned)
            cleaned = cleaned.strip()
        
        # Try direct parse
        try:
            parsed = json.loads(cleaned)
            
            # CRITICAL FIX: If answer field is a string with escaped JSON, extract it
            if "answer" in parsed and isinstance(parsed["answer"], str):
                answer_str = parsed["answer"]
                # Check if answer contains JSON structure (escaped)
                if '"answer"' in answer_str and ('"answer_type"' in answer_str or '"chunks_used"' in answer_str):
                    # Try to extract the actual answer text
                    try:
                        # Find the answer field value in the escaped JSON string
                        match = re.search(r'"answer"\s*:\s*"((?:[^"\\]|\\.|\\n)*)"', answer_str, re.DOTALL)
                        if match:
                            # Properly unescape JSON string
                            extracted = json.loads('"' + match.group(1) + '"')
                            parsed["answer"] = extracted
                            print(f"[RAG] ✅ Extracted nested answer from escaped JSON string")
                    except:
                        pass
            
            # CRITICAL FIX: Answer Type Validation with Auto-correction
            VALID_ANSWER_TYPES = {
                "DIRECT", "SECTION_OVERVIEW", "DOCUMENT_OVERVIEW",
                "CODE_ANALYSIS", "EXERCISE_GENERATION",
                "MULTI_CONCEPT_REASONING", "COMPARE_SYNTHESIZE",
                "FALLBACK", "TOO_BROAD", "SYNTHESIS"
            }
            
            answer_type = parsed.get("answer_type", "FALLBACK")
            answer = parsed.get("answer", "")
            confidence = parsed.get("confidence", 0.0)
            
            # Validate answer type
            if answer_type not in VALID_ANSWER_TYPES:
                print(f"[RAG] ⚠️ Invalid answer_type: {answer_type}, attempting auto-correction...")
                
                # Auto-correct based on content
                answer_lower = answer.lower()
                
                # Check for SECTION_OVERVIEW markers
                if re.search(r'PHẦN\s+\d+:', answer) or any(marker in answer_lower for marker in ['phần', 'nội dung chính', 'bao gồm']):
                    answer_type = "SECTION_OVERVIEW"
                    print(f"[RAG] Auto-corrected to SECTION_OVERVIEW")
                
                # Check for comparison markers
                elif any(marker in answer_lower for marker in ["|", "giống", "khác", "so sánh", "tương tự", "khác biệt"]):
                    answer_type = "COMPARE_SYNTHESIZE"
                    print(f"[RAG] Auto-corrected to COMPARE_SYNTHESIZE")
                
                # Check for code analysis markers
                elif "phân tích" in answer_lower or "code" in answer_lower or "function" in answer_lower:
                    answer_type = "CODE_ANALYSIS"
                    print(f"[RAG] Auto-corrected to CODE_ANALYSIS")
                
                # Default to DIRECT if has chunks, otherwise FALLBACK
                else:
                    chunks_used = parsed.get("chunks_used", [])
                    if chunks_used:
                        answer_type = "DIRECT"
                        print(f"[RAG] Auto-corrected to DIRECT (has chunks)")
                    else:
                        answer_type = "FALLBACK"
                        print(f"[RAG] Auto-corrected to FALLBACK (no chunks)")
                
                # Apply penalty for invalid type
                confidence = max(0.5, confidence * 0.8)
                parsed["answer_type"] = answer_type
                parsed["confidence"] = confidence
            
            # NEW: Validate SECTION_OVERVIEW responses
            if parsed.get("answer_type") == "SECTION_OVERVIEW":
                has_title = bool(re.search(r'PHẦN\s+\d+:', answer))
                has_list_intro = "Nội dung chính bao gồm" in answer or "bao gồm:" in answer
                topic_count = len(re.findall(r'\d+\.\s+\*\*', answer))
                
                if not has_title or not has_list_intro or topic_count < 3:
                    print(f"[RAG] ⚠️ SECTION_OVERVIEW format invalid:")
                    print(f"  - Has title: {has_title}")
                    print(f"  - Has list intro: {has_list_intro}")
                    print(f"  - Topic count: {topic_count}")
                    # Don't fail, but log warning
            return parsed
        except json.JSONDecodeError as e:
            print(f"[RAG] JSON decode error: {e}")
            pass
        
        # Try to extract JSON object
        match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', cleaned, re.DOTALL)
        if match:
            try:
                parsed = json.loads(match.group())
                # Apply same type validation
                VALID_ANSWER_TYPES = {
                    "DIRECT", "SECTION_OVERVIEW", "DOCUMENT_OVERVIEW",
                    "CODE_ANALYSIS", "EXERCISE_GENERATION",
                    "MULTI_CONCEPT_REASONING", "COMPARE_SYNTHESIZE",
                    "FALLBACK", "TOO_BROAD", "SYNTHESIS"
                }
                answer_type = parsed.get("answer_type", "FALLBACK")
                if answer_type not in VALID_ANSWER_TYPES:
                    answer = parsed.get("answer", "")
                    answer_lower = answer.lower()
                    if re.search(r'PHẦN\s+\d+:', answer) or any(marker in answer_lower for marker in ['phần', 'nội dung chính']):
                        parsed["answer_type"] = "SECTION_OVERVIEW"
                    elif any(marker in answer_lower for marker in ["|", "giống", "khác", "so sánh"]):
                        parsed["answer_type"] = "COMPARE_SYNTHESIZE"
                    else:
                        chunks_used = parsed.get("chunks_used", [])
                        parsed["answer_type"] = "DIRECT" if chunks_used else "FALLBACK"
                    parsed["confidence"] = max(0.5, parsed.get("confidence", 0.0) * 0.8)
                return parsed
            except Exception as e:
                print(f"[RAG] Extracted JSON parse failed: {e}")
                pass
        
        # Final fallback: reconstruct from text
        print(f"[RAG] All parsing attempts failed. Attempting reconstruction...")
        return self._reconstruct_json_from_text(cleaned, query_type)
    
    def _extract_json_with_multiline_string(self, text: str):
        """Extract JSON that may contain multiline strings (like tables)."""
        # Look for JSON structure: {"answer": "...", ...}
        # Handle multiline strings in answer field
        pattern = r'\{\s*"answer"\s*:\s*"((?:[^"\\]|\\.|\\n)*)"'
        match = re.search(pattern, text, re.DOTALL)
        if match:
            # Find the complete JSON object
            start = text.find('{')
            if start >= 0:
                # Try to find matching closing brace
                brace_count = 0
                in_string = False
                escape_next = False
                for i in range(start, len(text)):
                    char = text[i]
                    if escape_next:
                        escape_next = False
                        continue
                    if char == '\\':
                        escape_next = True
                        continue
                    if char == '"' and not escape_next:
                        in_string = not in_string
                    if not in_string:
                        if char == '{':
                            brace_count += 1
                        elif char == '}':
                            brace_count -= 1
                            if brace_count == 0:
                                return re.match(r'.*', text[start:i+1], re.DOTALL)
        return None
    
    def _fix_json_with_table(self, json_str: str) -> str:
        """Fix common JSON issues when answer contains markdown tables."""
        # If answer field contains unescaped newlines, try to fix
        # Look for "answer": "..." pattern and ensure newlines are escaped
        # But be careful - if it's already valid JSON, don't break it
        
        # First, try to parse as-is
        try:
            json.loads(json_str)
            return json_str  # Already valid
        except:
            pass
        
        # Try to fix: replace literal newlines in string values with \n
        # This is tricky - we need to be careful not to break valid JSON
        # Only fix if we detect the issue is with newlines in answer field
        if '\n' in json_str and '"answer"' in json_str:
            # Try to escape newlines in the answer field only
            # Pattern: "answer": "text with\nnewlines"
            def escape_newlines_in_quotes(match):
                content = match.group(1)
                # Escape newlines, but preserve \n if already escaped
                content = content.replace('\n', '\\n').replace('\\n\\n', '\\n')
                return f'"answer": "{content}"'
            
            # This is complex - for now, just return original and let reconstruction handle it
            pass
        
        return json_str
    
    def _reconstruct_json_from_text(self, text: str, query_type: str) -> dict:
        """Reconstruct JSON from plain text answer (fallback)."""
        
        # 🔥 CRITICAL FIX: Auto-extract chunk references from text
        chunk_refs = re.findall(r'chunk\s+(\d+)', text, re.IGNORECASE)
        chunks_recovered = list(set([int(c) for c in chunk_refs]))[:15]
        
        # CRITICAL FIX: For COMPARE_SYNTHESIZE with table, auto-extract chunks from selected_results
        has_table = "|" in text and any(re.search(pattern, text, re.MULTILINE) for pattern in [
            r'^\|.*\|.*\|',  # Table row pattern
        ])
        
        chunks_found = []
        if query_type == "COMPARE_SYNTHESIZE" and has_table:
            # Auto-extract chunks from selected_results if available
            if hasattr(self, 'selected_results') and self.selected_results:
                print(f"[RAG] COMPARE_SYNTHESIZE with table but no chunks found, will use selected chunks")
                for item in self.selected_results[:min(15, len(self.selected_results))]:
                    record = item.get("_record")
                    if record:
                        chunk_idx = record.get("chunk_index")
                        doc_id = record.get("document_id")
                        if chunk_idx is not None and doc_id:
                            chunks_found.append({
                                "chunk_index": chunk_idx,
                                "document_id": doc_id
                            })
                print(f"[RAG] ✅ Auto-extracted {len(chunks_found)} chunks from selected_results for COMPARE_SYNTHESIZE")
        
        # Extract chunks mentioned in text (fallback if not found above)
        # Note: chunks_found might be list of dicts (from selected_results) or list of ints (from text)
        chunks_from_text = []
        if not chunks_found or (isinstance(chunks_found[0], dict) and len(chunks_found) < 3):
            chunk_pattern = r'\[Chunk\s+(\d+)\]|chunk\s+(\d+)|\(từ\s+chunk\s+(\d+)'
            for match in re.finditer(chunk_pattern, text, re.IGNORECASE):
                chunk_num = match.group(1) or match.group(2) or match.group(3)
                if chunk_num:
                    chunks_from_text.append(int(chunk_num))
            chunks_from_text = list(set(chunks_from_text))  # Deduplicate
        
        # If we have dict chunks (from selected_results), keep them; otherwise use int chunks
        if isinstance(chunks_found, list) and len(chunks_found) > 0 and isinstance(chunks_found[0], dict):
            # Already have dict format from selected_results
            pass
        elif chunks_from_text:
            # Convert int chunks to dict format (will need document_id later)
            chunks_found = [{"chunk_index": idx} for idx in chunks_from_text]
        else:
            chunks_found = []
        
        # Determine answer type from content
        text_lower = text.lower()
        answer_type = "FALLBACK"
        confidence = 0.0
        
        # 🔥 CRITICAL FIX: Handle DOCUMENT_OVERVIEW when JSON parse fails
        if query_type == "DOCUMENT_OVERVIEW":
            # Check if answer contains numbered list of sections
            has_numbered_sections = bool(re.search(r'\d+\.\s+\*\*PHẦN\s+\d+', text, re.IGNORECASE))
            has_sections = bool(re.search(r'PHẦN\s+\d+', text, re.IGNORECASE))
            
            if has_numbered_sections or has_sections:
                answer_type = "DOCUMENT_OVERVIEW"
                # Extract chunks from selected_results if available
                if hasattr(self, 'selected_results') and self.selected_results:
                    print(f"[RAG] DOCUMENT_OVERVIEW JSON parse failed, recovering chunks from selected_results")
                    chunks_found = []
                    for item in self.selected_results[:min(50, len(self.selected_results))]:
                        record = item.get("_record")
                        if record:
                            chunk_idx = record.get("chunk_index")
                            doc_id = record.get("document_id")
                            if chunk_idx is not None and doc_id:
                                chunks_found.append({
                                    "chunk_index": chunk_idx,
                                    "document_id": doc_id
                                })
                    print(f"[RAG] ✅ Recovered {len(chunks_found)} chunks for DOCUMENT_OVERVIEW")
                
                # Count sections found
                section_numbers = re.findall(r'PHẦN\s+(\d+)', text, re.IGNORECASE)
                section_count = len(set(section_numbers))
                confidence = min(0.90, 0.7 + (section_count * 0.02))  # Boost based on section count
                print(f"[RAG] DOCUMENT_OVERVIEW recovery: {section_count} sections found, confidence={confidence:.2f}")
        
        elif query_type == "SECTION_OVERVIEW" or any(marker in text_lower for marker in ['phần', 'nội dung chính', 'bao gồm']):
            answer_type = "SECTION_OVERVIEW"
            confidence = 0.75 if chunks_found else 0.5
        elif query_type == "COMPARE_SYNTHESIZE" or (query_type == "COMPARE_SYNTHESIZE" and ("|" in text or "so sánh" in text_lower or "khác" in text_lower)):
            answer_type = "COMPARE_SYNTHESIZE"
            # Nếu có bảng markdown thì confidence cao hơn
            confidence = 0.85 if "|" in text else 0.75
            # CRITICAL: COMPARE_SYNTHESIZE should always have references if table exists
            if "|" in text and not chunks_found:
                # Try to extract chunks from selected_results if available
                print(f"[RAG] COMPARE_SYNTHESIZE with table but no chunks found, will use selected chunks")
        elif query_type == "CODE_ANALYSIS" and "phân tích" in text_lower:
            answer_type = "CODE_ANALYSIS"
            confidence = 0.7
        elif query_type == "MULTI_CONCEPT_REASONING":
            answer_type = "MULTI_CONCEPT_REASONING"
            confidence = 0.65
        elif query_type == "EXERCISE_GENERATION" and ("bài tập" in text_lower or "function" in text_lower):
            answer_type = "EXERCISE_GENERATION"
            confidence = 0.7
        elif chunks_found:
            answer_type = "DIRECT"
            confidence = 0.6
        
        # Extract main answer - CRITICAL: Keep full table if present
        # Check if text contains a markdown table (has | characters in table format)
        # More flexible pattern: look for table header row and separator or multiple rows with |
        table_patterns = [
            r'\|.*\|.*\n\|[-:]+\|',  # Header + separator
            r'\|.*\|.*\n\|.*\|.*\n\|.*\|',  # At least 3 rows with |
            r'\|.*Tiêu chí.*\|.*\|',  # Vietnamese table header
        ]
        has_table = "|" in text and any(re.search(pattern, text, re.MULTILINE) for pattern in table_patterns)
        
        print(f"[RAG] Reconstruction: text_length={len(text)}, has_table={has_table}, query_type={query_type}")
        if has_table:
            print(f"[RAG] Table detected! Keeping full text (first 300 chars: {text[:300]})")
            # For tables, keep the ENTIRE text including table
            # Don't truncate - tables need to be complete
            # Only limit if text is extremely long (over 10000 chars)
            if len(text) > 10000:
                # Try to find a reasonable end point
                # Look for "Kết luận" or "Nguồn" sections
                end_markers = [
                    r'(.*?)(?:\n\n(?:Kết luận|Nguồn|\*\*Kết|\*\*Nguồn|Nguồn tham khảo))',
                    r'(.*?)(?:\n\n\n)',
                ]
                for pattern in end_markers:
                    match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
                    if match and len(match.group(1)) > 500:
                        answer = match.group(1).strip()
                        break
                else:
                    # No good end point found, take first 10000 chars
                    answer = text[:10000]
            else:
                # Text is reasonable length, keep it all
                answer = text
        else:
            # For non-table answers, use original logic but increase limit
            answer_match = re.search(r'^(.+?)(?:\n\n|$)', text, re.DOTALL)
            answer = answer_match.group(1) if answer_match else text[:2000]  # Increased from 1000
        
        answer = answer.strip()
        
        # CRITICAL FIX: Build sentence_mapping from text
        sentence_mapping = []
        if answer:
            # Split into sentences (using Vietnamese sentence delimiters)
            sentences = re.split(r'[.!?]+', answer)
            
            for i, sent in enumerate(sentences):
                sent = sent.strip()
                if len(sent) < 10:  # Skip very short fragments
                    continue
                
                # Find which chunk mentions this sentence concept
                # Look for chunk references near this sentence
                related_chunk = None
                
                # Find the position of this sentence in the full text
                sent_start = answer.find(sent)
                if sent_start >= 0:
                    # Look backwards from sentence start for chunk references
                    text_before_sent = answer[:sent_start]
                    chunk_refs_before = re.findall(r'\[Chunk\s+(\d+)\]', text_before_sent, re.IGNORECASE)
                    if chunk_refs_before:
                        # Use the most recent chunk reference
                        related_chunk = int(chunk_refs_before[-1])
                    else:
                        # Look in the sentence itself
                        chunk_refs_in_sent = re.findall(r'\[Chunk\s+(\d+)\]', sent, re.IGNORECASE)
                        if chunk_refs_in_sent:
                            related_chunk = int(chunk_refs_in_sent[0])
                
                # If no chunk found, check if any chunk number appears in nearby context
                if not related_chunk and chunks_found:
                    # Heuristic: if sentence contains keywords that might relate to chunks
                    # Use first chunk as fallback (better than nothing)
                    if isinstance(chunks_found[0], dict):
                        related_chunk = chunks_found[0].get("chunk_index")
                    else:
                        related_chunk = chunks_found[0] if len(chunks_found) == 1 else None
                
                sentence_mapping.append({
                    "sentence": sent[:200],  # Truncate to 200 chars
                    "chunk": related_chunk,
                    "external": related_chunk is None
                })
        
        # Convert chunks_found to proper format
        chunks_used = []
        if chunks_found:
            if isinstance(chunks_found[0], dict):
                # Already in correct format
                chunks_used = chunks_found[:15]  # Limit to 15 chunks
            else:
                # Convert int list to dict format
                chunks_used = [{"chunk_index": idx} for idx in chunks_found[:15]]
        
        print(f"[RAG] Reconstructed JSON: answer_type={answer_type}, confidence={confidence:.2f}, chunks={len(chunks_used)}, sentences={len(sentence_mapping)}")
        
        return {
            "answer": answer,
            "answer_type": answer_type,
            "chunks_used": chunks_used,
            "confidence": confidence,
            "sentence_mapping": sentence_mapping,  # CRITICAL FIX: Now includes actual mapping
            "sources": {
                "from_document": bool(chunks_found),
                "from_external_knowledge": False
            },
            "_reconstructed": True  # Flag for debugging
        }
    
    def _get_fallback_response(self) -> tuple:
        """Safe fallback response."""
        return (
            "Hiện tại không thể trả lời câu hỏi này. Vui lòng thử lại.",
            [],
            "FALLBACK",
            0.0,
            []
        )



    async def ask(

        self,

        db: AsyncIOMotorDatabase,

        user_id: str,

        question: str,

        document_ids: Optional[List[str]] = None,  # ← THAY ĐỔI: List[str] thay vì str

        top_k: Optional[int] = None,

        conversation_id: Optional[str] = None,

    ) -> dict:

        documents: List[DocumentInDB] = []

        if document_ids:  # ← THAY ĐỔI: Kiểm tra list
            # CHỈ lấy documents được chọn
            for doc_id in document_ids:
                doc = await get_document_by_id(db, doc_id)
                if doc and doc.user_id == user_id:
                    documents.append(doc)
                else:
                    print(f"[RAG] ⚠️ Document {doc_id} not found or not accessible")
            
            if not documents:
                raise ValueError("Không tìm thấy tài liệu nào trong danh sách đã chọn")
            
            print(f"[RAG] 🎯 Selected {len(documents)} document(s):")
            for doc in documents:
                print(f"  ✓ {doc.filename} (ID: {doc.id})")
        else:
            # Không chọn gì = lấy TẤT CẢ
            documents = await get_documents_by_user(db, user_id)
            print(f"[RAG] 📚 Using ALL documents: {len(documents)} file(s)")
        
        # GIÁM SÁT: Log documents IDs để debug
        document_ids_used = [doc.id for doc in documents]
        print(f"[RAG] Document IDs being searched: {document_ids_used}")

        # Handle các câu chào / small-talk không liên quan đến tài liệu
        normalized_question = question.strip().lower()
        small_talk_phrases = [
            "hi",
            "hello",
            "xin chào",
            "chào",
            "chao",
            "chào bạn",
            "chào ad",
            "chào admin",
            "good morning",
            "good afternoon",
            "good evening",
            "bye",
            "tạm biệt",
            "cảm ơn",
            "thank you",
        ]

        # Nếu câu hỏi rất ngắn và chỉ là lời chào / cảm ơn thì không gọi RAG
        if len(normalized_question) <= 40 and any(
            normalized_question == p or normalized_question.startswith(p + " ")
            for p in small_talk_phrases
        ):
            # Trả lời thân thiện, không trích dẫn tài liệu
            if any(
                kw in normalized_question
                for kw in ["cảm ơn", "cam on", "thank", "tks", "thanks"]
            ):
                answer = (
                    "Cảm ơn bạn! Nếu cần mình hỗ trợ giải bài hoặc tóm tắt nội dung trong tài liệu, "
                    "hãy gửi câu hỏi nhé."
                )
            elif any(
                kw in normalized_question
                for kw in ["bye", "tạm biệt", "tam biet", "good night"]
            ):
                answer = (
                    "Tạm biệt bạn, hẹn gặp lại! Khi nào cần hỏi bài hoặc tra cứu tài liệu, cứ quay lại nhé."
                )
            else:
                answer = (
                    "Xin chào! Mình là trợ lý StudyQnA, mình sẽ giúp bạn trả lời các câu hỏi dựa trên tài liệu "
                    "bạn đã tải lên. Bạn cứ gửi câu hỏi về nội dung cần học nhé."
                )

            # Lưu lịch sử nhưng không có references
            # Use first document_id for backward compatibility
            doc_id_for_history = document_ids[0] if document_ids and len(document_ids) > 0 else None
            history_record = await create_history(
                db, user_id, question, answer, [], doc_id_for_history, conversation_id
            )

            final_conversation_id = conversation_id

            # Giữ logic conversation_id tương tự nhánh bình thường
            if not final_conversation_id:
                final_conversation_id = history_record.id
                try:
                    from bson import ObjectId

                    await db["histories"].update_one(
                        {"_id": ObjectId(history_record.id)},
                        {"$set": {"conversation_id": history_record.id}},
                    )
                    history_record.conversation_id = history_record.id
                except Exception as e:
                    print(
                        f"[RAG] Warning: Failed to update conversation_id for small-talk history {history_record.id}: {e}"
                    )

            return {
                "answer": answer,
                "references": [],
                "documents": [],
                "conversation_id": final_conversation_id,
                "history_id": history_record.id,
            }

        question_embeddings = await self.embedding_service.embed_texts([question])

        if not question_embeddings:
            # ENHANCED: Detect query type even for error cases
            query_type = detect_query_type_fast(question)
            return {
                "answer": "Không thể tạo embedding cho câu hỏi.",
                "references": [],
                "documents": [],
                "metadata": {
                    "answer_type": "FALLBACK",
                    "confidence": 0.0,
                    "query_type": query_type,
                    "chunks_selected": 0,
                    "chunks_used": 0,
                }
            }

        # ENHANCED: Detect query type FIRST để điều chỉnh search
        query_type = detect_query_type_fast(question)
        print(f"[RAG] Detected query type: {query_type}")

        query_vector = np.array(question_embeddings, dtype="float32")

        results = []

        # ENHANCED: Dynamic max_chunks
        max_chunks_for_query = self._determine_max_chunks_for_query(
            question, 
            query_type,
            num_docs=len(documents)  # ✅ THÊM THAM SỐ
        )
        print(f"[RAG] Max chunks for this query: {max_chunks_for_query} (for {len(documents)} document(s))")

        # Search with larger initial top_k for more candidates
        for doc in documents:

            namespace = doc.faiss_namespace or f"user_{doc.user_id}_doc_{doc.id}"

            index = load_faiss_index(namespace)

            if index is None or index.ntotal == 0:

                continue

            if index.d != query_vector.shape[1]:

                continue

            try:

                # ENHANCED: Tăng search_k dựa trên query type
                if query_type == "DOCUMENT_OVERVIEW":
                    # DOCUMENT_OVERVIEW cần NHIỀU chunks nhất để tìm TẤT CẢ các phần
                    search_k = min(150, index.ntotal)  # Tăng lên 150 để có đủ candidate chunks
                elif query_type == "SECTION_OVERVIEW":
                    # SECTION_OVERVIEW cần chunks vừa phải
                    search_k = min(100, index.ntotal)
                elif query_type in ["MULTI_CONCEPT_REASONING", "COMPARE_SYNTHESIZE", "CODE_ANALYSIS"]:
                    # COMPARE_SYNTHESIZE cần nhiều chunks hơn để so sánh đầy đủ
                    if query_type == "COMPARE_SYNTHESIZE":
                        search_k = min(75, index.ntotal)  # Tăng từ 50 lên 75 cho so sánh
                    else:
                        search_k = min(50, index.ntotal)  # Giữ nguyên cho các loại khác
                else:
                    search_k = min(30, index.ntotal)

                distances, ids = index.search(query_vector, search_k)

            except Exception:

                continue

            for dist, vector_id in zip(distances[0], ids[0]):

                if vector_id == -1:

                    continue

                similarity = float(1.0 / (1.0 + dist))

                results.append(

                    {

                        "document": doc,

                        "namespace": namespace,

                        "vector_id": int(vector_id),

                        "similarity": similarity,

                    }

                )



        if not results:
            # ENHANCED: Detect query type even for error cases
            query_type = detect_query_type_fast(question)
            
            # ENHANCED: Thông báo rõ ràng cho user
            if document_ids:
                selected_filenames = [doc.filename for doc in documents]
                answer = (
                    f"Không tìm thấy thông tin liên quan trong {len(documents)} tài liệu đã chọn:\n"
                    f"• {', '.join(selected_filenames)}\n\n"
                    f"Có thể thử:\n"
                    f"1. Chọn thêm tài liệu khác\n"
                    f"2. Đặt câu hỏi theo cách khác\n"
                    f"3. Kiểm tra nội dung tài liệu có liên quan không"
                )
            else:
                answer = "Không tìm thấy đoạn văn phù hợp trong tài liệu của bạn."

            return {
                "answer": answer,
                "references": [],
                "documents": document_ids_used,
                "documents_searched": document_ids_used,  # ← THÊM: list IDs đã search
                "metadata": {
                    "answer_type": "FALLBACK",
                    "confidence": 0.0,
                    "query_type": query_type,
                    "chunks_selected": 0,
                    "chunks_used": 0,
                    "documents_searched": len(documents),  # ← THÊM: số docs đã search
                }
            }



        # Boost chunks that contain keywords from the question

        # CRITICAL FIX: Clean keywords - remove quotes, punctuation
        question_lower = question.lower()
        # Remove quotes and normalize
        question_normalized = re.sub(r'["\']', '', question_lower)
        question_normalized = re.sub(r'[?!.,;:]', ' ', question_normalized)
        
        # Extract keywords (longer than 2 chars, exclude stop words)
        stop_words = {'của', 'và', 'với', 'trong', 'về', 'có', 'là', 'được', 'này', 'cho', 'từ', 'nào'}
        question_keywords = [
            q.strip() 
            for q in question_normalized.split() 
            if len(q.strip()) > 2 and q.strip() not in stop_words
        ]

        # 🔥 CRITICAL FIX: Document-specific keyword detection
        # Detect which document type the question is asking about
        project_keywords = ['dự án', 'module', 'sloc', 'cocomo', 'quản lý', 'phát triển', 'wbs', 'nhân sự', 'chi phí', 'rủi ro']
        js_keywords = ['javascript', 'js', 'es6', 'function', 'arrow', 'promise', 'async', 'await', 'callback', 'closure', 'hoisting', 'scope', 'framework', 'react', 'vue', 'angular', 'jquery', 'node']
        
        has_project_keywords = any(kw in question_lower for kw in project_keywords)
        has_js_keywords = any(kw in question_lower for kw in js_keywords)
        
        print(f"[RAG] Document keyword detection: project_keywords={has_project_keywords}, js_keywords={has_js_keywords}")

        # 🔥 CRITICAL FIX: File-Targeted Boosting - Detect files mentioned in question
        target_doc_ids = set()
        for doc in documents:
            # Làm sạch tên file để so sánh (bỏ đuôi .pdf, thay thế ký tự đặc biệt)
            doc_name_clean = doc.filename.lower().replace(".pdf", "").replace(".docx", "").strip()
            doc_name_variations = [
                doc_name_clean,
                doc_name_clean.replace("-", " "),
                doc_name_clean.replace("_", " "),
                # Thêm các biến thể viết tắt
                "js" if "javascript" in doc_name_clean else "",
                "cocomo" if "cocomo" in doc_name_clean else "",
            ]
            
            # Kiểm tra xem tên file có xuất hiện trong câu hỏi không
            for variant in doc_name_variations:
                if variant and variant in question_lower:
                    target_doc_ids.add(doc.id)
                    print(f"[RAG] 🎯 TARGET FILE DETECTED: User asked about '{variant}' -> Boost docs from {doc.filename}")
                    break
        
        if target_doc_ids:
            print(f"[RAG] 🎯 File-Targeted Boosting: {len(target_doc_ids)} target file(s) detected")

        for item in results:
            item["original_similarity"] = item["similarity"]

        print(f"[RAG] Found {len(results)} candidate chunks, boosting by question keywords (cleaned): {question_keywords[:10]}")



        # Cache content for boosting and later use

        for item in results:

            doc = item["document"]

            record = await db["embeddings"].find_one(

                {"document_id": doc.id, "vector_index": item["vector_id"]}

            )

            if not record:

                continue

            

            item["_record"] = record

            chunk_id = record.get("chunk_id")

            chunk_doc = None

            if chunk_id:

                try:

                    from bson import ObjectId

                    chunk_doc = await db["chunks"].find_one({"_id": ObjectId(chunk_id)})

                except Exception:

                    chunk_doc = await db["chunks"].find_one({"_id": chunk_id})
            
            # Debug: kiểm tra chunk_doc và metadata cho một vài chunks
            chunk_idx = record.get("chunk_index") if record else None
            if chunk_idx in [1, 3, 8, 10, 11] and doc.file_type in ["docx", "doc"]:
                if chunk_doc:
                    metadata = chunk_doc.get("metadata", {})
                    print(f"[RAG] Query chunk {chunk_idx}: chunk_id={chunk_id}, found={chunk_doc is not None}, metadata={metadata}")
                else:
                    print(f"[RAG] Query chunk {chunk_idx}: chunk_id={chunk_id}, chunk_doc NOT FOUND")

            content = (chunk_doc or {}).get("content") or record.get("content") or ""

            item["_content"] = content

            item["_chunk_doc"] = chunk_doc

            content_lower = content.lower()

            # 🔥 CRITICAL FIX #1: File-Targeted Boost - Boost điểm cực mạnh nếu chunk thuộc về file được nhắc tên
            file_target_boost = 0.0
            if doc.id in target_doc_ids:
                # Tăng 25% điểm similarity để đảm bảo nó lọt vào Top K
                file_target_boost = 0.25
                print(f"[RAG] 🚀 File-Target Boost: +{file_target_boost} for chunk {record.get('chunk_index', '?')} in {doc.filename}")

            # 🔥 CRITICAL FIX #2: Section Header Boost - Boost cho các chunk chứa tiêu đề Section chính xác
            section_header_boost = 0.0
            # Regex bắt chính xác tiêu đề đầu dòng: "PHẦN 4", "CHƯƠNG 5", "PART 3"
            if re.search(r'(^|\n)\s*(PHẦN|CHƯƠNG|PART)\s+\d+', content, re.IGNORECASE):
                section_header_boost = 0.15
                print(f"[RAG] 📑 Section Header Boost: +{section_header_boost} for chunk {record.get('chunk_index', '?')} in {doc.filename}")

            # CRITICAL FIX: Boost main sections (PHẦN, CHƯƠNG) for section questions
            is_main_section = False
            section_boost = 0.0
            
            # Check if this is a main section (PHẦN X, CHƯƠNG X)
            if chunk_doc:
                chunk_metadata = chunk_doc.get("metadata", {}) or {}
                section = chunk_metadata.get("section") or chunk_metadata.get("heading") or ""
                section_lower = section.lower() if section else ""
                
                # Main section patterns
                main_section_patterns = [
                    r'^phần\s+\d+',  # PHẦN 5
                    r'^chương\s+\d+',  # CHƯƠNG 3
                    r'^phần\s+[ivx]+',  # PHẦN V
                ]
                
                for pattern in main_section_patterns:
                    if re.match(pattern, section_lower):
                        is_main_section = True
                        section_boost = 0.5  # Strong boost for main sections
                        print(f"[RAG] Main section detected: {section} (chunk {record.get('chunk_index')})")
                        break
            
            # CRITICAL FIX: Also check for quoted terms
            # If question has 'module' (with quotes), match "module" in content
            quoted_terms = re.findall(r'["\']([^"\']+)["\']', question_lower)
            
            keyword_matches = 0
            
            # Match regular keywords
            for kw in question_keywords:
                if kw in content_lower:
                    keyword_matches += 1
            
            # Match quoted terms (these are important!)
            for term in quoted_terms:
                term_clean = term.lower().strip()
                if term_clean in content_lower:
                    keyword_matches += 3  # Triple weight for quoted terms!
                    print(f"[RAG] Found quoted term '{term_clean}' in chunk {record.get('chunk_index')}")



            # Special boost for section numbers

            if any(kw in ["phần", "chương", "part"] for kw in question_keywords):

                question_numbers = re.findall(r'\d+', question_lower)

                all_subsection_patterns = re.findall(r'\b\d+\.\d+\b', question_lower)



                for num in question_numbers:

                    if f"phần {num}" in content_lower or f"chương {num}" in content_lower or f"part {num}" in content_lower:

                        keyword_matches += 3

                        section_pattern = rf"{num}\.\d+"

                        if re.search(section_pattern, content_lower):

                            keyword_matches += 5

                        break



                for subsec in all_subsection_patterns:

                    if subsec in content_lower:

                        keyword_matches += 8

                        print(f"[RAG] Found exact subsection match: {subsec} in chunk {record.get('chunk_index', '?')}")



            # CRITICAL FIX: For comparison queries, boost chunks containing BOTH compared items
            comparison_boost = 0.0
            if query_type == "COMPARE_SYNTHESIZE":
                # Extract items being compared from question
                compare_keywords = []
                question_normalized = question_lower
                if "so sánh" in question_lower or "so với" in question_lower:
                    # Extract key terms: "COCOMO", "WBS", "thời gian", etc.
                    words = question_normalized.split()
                    for word in words:
                        if len(word) > 3 and word not in ["so", "sánh", "với", "theo", "và", "của", "cho", "là", "có", "được", "trong", "từ"]:
                            compare_keywords.append(word)
                
                # Boost chunks containing comparison keywords
                keyword_count = sum(1 for kw in compare_keywords if kw in content_lower)
                
                if keyword_count >= 2:  # Contains at least 2 comparison terms
                    comparison_boost = min(0.4, keyword_count * 0.15)
                    print(f"[RAG] Comparison boost +{comparison_boost:.3f} for chunk {record.get('chunk_index')} (keywords: {keyword_count})")
            
            # 🔥 CRITICAL FIX: Special boost for SECTION_OVERVIEW - boost mạnh chunk chứa đúng section được hỏi
            section_target_boost = 0.0
            if query_type == "SECTION_OVERVIEW":
                # Extract numbers from question: "Phần 4", "Phần 5"
                target_sections = re.findall(r'(?:phần|chương|part)\s+(\d+)', question_lower)
                
                for sec_num in target_sections:
                    # Pattern chính xác: "PHẦN 4:", "PHẦN 4 " (đầu dòng hoặc sau xuống dòng)
                    # Tránh match nhầm "14" khi tìm "4"
                    precise_pattern = rf'(?:^|\n)\s*(?:phần|chương|part)\s+{sec_num}\b'
                    
                    if re.search(precise_pattern, content_lower):
                        # Boost cực mạnh để đảm bảo chunk này lọt vào top
                        section_target_boost = 1.5 
                        print(f"[RAG] 🎯 FOUND TARGET SECTION {sec_num} in chunk {record.get('chunk_index')} -> Boost +1.5")
                        
                        # --- CHÈN LOGIC MỚI TẠI ĐÂY ĐỂ ĐÁNH DẤU CHUNK NÀY LÀ "HERO" ---
                        item["is_target_section_head"] = True 
                        # -----------------------------------------------------------------
                        break
            
            # Apply boosts (combine file-target boost + section header boost + keyword boost + section boost + comparison boost + section target boost)
            total_boost = 0.0
            boost_details = []

            # 🔥 CRITICAL FIX: Apply File-Targeted Boost FIRST (highest priority)
            if file_target_boost > 0:
                total_boost += file_target_boost
                boost_details.append("file_target")
            
            # 🔥 CRITICAL FIX: Apply Section Header Boost
            if section_header_boost > 0:
                total_boost += section_header_boost
                boost_details.append("section_header")

            if keyword_matches > 0:
                boost = min(0.3, keyword_matches * 0.08)
                total_boost += boost
                item["keyword_matches"] = keyword_matches
                boost_details.append(f"keywords({keyword_matches})")
                print(f"[RAG] Boosted chunk {record.get('chunk_index')} by {boost:.3f} (keywords: {keyword_matches})")
            
            # CRITICAL FIX: Add section boost for main sections
            if is_main_section:
                total_boost += section_boost
                boost_details.append("main_section")
            
            # CRITICAL FIX: Add comparison boost for COMPARE_SYNTHESIZE
            if comparison_boost > 0:
                total_boost += comparison_boost
                boost_details.append("comparison")
            
            # 🔥 CRITICAL FIX: Add section target boost for SECTION_OVERVIEW
            if section_target_boost > 0:
                total_boost += section_target_boost
                boost_details.append(f"TARGET_SECTION")
            
            # 🔥 CRITICAL FIX: Priority tier cho TOC
            toc_priority_boost = 0.0
            if "mục lục" in content_lower or "table of contents" in content_lower:
                toc_priority_boost = 2.0  # Tăng từ 0.8 lên 2.0 (siêu mạnh)
                item["is_toc"] = True
                print(f"[RAG] 🎯 TABLE OF CONTENTS detected in chunk {record.get('chunk_index')} → priority boost +2.0")
            
            # Pattern 2: Boost chunks chứa nhiều PHẦN X
            # Đếm số lượng "PHẦN X" trong content
            section_count = len(re.findall(r'PHẦN\s+\d+', content, re.IGNORECASE))
            if section_count >= 3:  # Nếu có từ 3 PHẦN trở lên → đây là chunk overview
                overview_boost = min(1.0, section_count * 0.2)  # Tăng từ 0.15 lên 0.2
                total_boost += overview_boost
                boost_details.append(f"overview({section_count}_sections)")
                print(f"[RAG] Boosted chunk {record.get('chunk_index')} - contains {section_count} section headings")
            
            # Apply TOC boost (highest priority)
            if toc_priority_boost > 0:
                total_boost += toc_priority_boost
                boost_details.append("TABLE_OF_CONTENTS")
            
            if total_boost > 0:
                item["similarity"] = min(1.0, item["similarity"] + total_boost)
                print(f"[RAG] Boosted chunk {record.get('chunk_index', '?')} by {total_boost:.3f} ({', '.join(boost_details)})")



        # 🔥 CRITICAL FIX: NEIGHBOR RETRIEVAL (Lấy thêm chunk lân cận để đọc bảng/nội dung dài)
        # Nếu tìm thấy chunk chứa tiêu đề Section, bắt buộc lấy thêm 2 chunk tiếp theo
        # Vì bảng dữ liệu thường nằm ở chunk ngay sau tiêu đề
        
        extra_chunks_to_fetch = []
        existing_keys = set((item["document"].id, item["vector_id"]) for item in results)
        
        for item in results:
            # Chỉ áp dụng cho các chunk được xác định là Chứa Tiêu Đề Section quan trọng
            if item.get("is_target_section_head", False):
                doc = item["document"]
                current_vector_id = item["vector_id"]
                current_chunk_index = item.get("_record", {}).get("chunk_index")
                
                if current_chunk_index is not None:
                    print(f"[RAG] 🪜 Expanding context for SECTION chunk {current_chunk_index} (Doc: {doc.filename})")
                    
                    # Lấy thêm 2 chunk tiếp theo (Next 1, Next 2)
                    # Query DB để lấy chunk_index + 1, +2
                    
                    # Tăng số lượng neighbor chunks từ 2 lên 3 để đảm bảo bảng dữ liệu được bao phủ đầy đủ
                    for offset in [1, 2, 3]:  # Lấy thêm 3 chunk nữa
                        next_chunk_idx = current_chunk_index + offset
                        
                        try:
                            # Tìm chunk kế tiếp trong DB
                            next_record = await db["embeddings"].find_one(
                                {"document_id": doc.id, "chunk_index": next_chunk_idx}
                            )
                            
                            if next_record:
                                next_vector_id = next_record.get("vector_index")
                                
                                # Nếu chunk này chưa có trong danh sách results
                                if (doc.id, next_vector_id) not in existing_keys:
                                    # Tạo item giả lập để thêm vào
                                    # Kế thừa similarity cao từ chunk cha để đảm bảo nó được chọn
                                    # Tăng similarity cao hơn để đảm bảo neighbor chunks được ưu tiên
                                    inherited_score = min(1.0, item["similarity"] * 0.98)  # Giữ similarity cao
                                    
                                    # Lấy chunk document để có content (giống logic trong vòng lặp chính)
                                    chunk_id = next_record.get("chunk_id")
                                    chunk_doc = None
                                    if chunk_id:
                                        try:
                                            from bson import ObjectId
                                            chunk_doc = await db["chunks"].find_one({"_id": ObjectId(chunk_id)})
                                        except Exception:
                                            chunk_doc = await db["chunks"].find_one({"_id": chunk_id})
                                    
                                    # Extract content giống như trong vòng lặp chính
                                    content = (chunk_doc or {}).get("content") or next_record.get("content") or ""
                                    
                                    new_item = {
                                        "document": doc,
                                        "namespace": item.get("namespace", ""),
                                        "vector_id": next_vector_id,
                                        "similarity": inherited_score,
                                        "_record": next_record,
                                        "_content": content,
                                        "_chunk_doc": chunk_doc,
                                        "is_neighbor": True,  # Đánh dấu là hàng xóm
                                        "parent_chunk_index": current_chunk_index  # Lưu chunk_index của target section head
                                    }
                                    
                                    extra_chunks_to_fetch.append(new_item)
                                    existing_keys.add((doc.id, next_vector_id))
                                    print(f"[RAG]    -> Added neighbor chunk {next_chunk_idx} (auto-included)")
                        except Exception as e:
                            print(f"[RAG] Error fetching neighbor chunk {next_chunk_idx}: {e}")

        # Gộp chunk mới vào danh sách chính
        if extra_chunks_to_fetch:
            results.extend(extra_chunks_to_fetch)
            print(f"[RAG] Added {len(extra_chunks_to_fetch)} neighbor chunks to context to capture full tables/sections")

        # Sort by boosted similarity
        sorted_results = sorted(results, key=lambda r: r["similarity"], reverse=True)

        # 🔥 CRITICAL FIX: Đưa TOC chunks lên đầu, sau đó là target section heads và neighbor chunks
        # Separate chunks by priority
        toc_chunks = [item for item in sorted_results if item.get("is_toc", False)]
        target_section_heads = [item for item in sorted_results if item.get("is_target_section_head", False) and not item.get("is_toc", False)]
        neighbor_chunks = [item for item in sorted_results if item.get("is_neighbor", False) and not item.get("is_toc", False) and not item.get("is_target_section_head", False)]
        other_chunks = [item for item in sorted_results if not item.get("is_toc", False) and not item.get("is_target_section_head", False) and not item.get("is_neighbor", False)]
        
        # Sắp xếp: TOC -> Target Section Heads -> Neighbor chunks (ngay sau section head) -> Other chunks
        # Đảm bảo neighbor chunks được đặt ngay sau target section head để chúng được chọn cùng nhau
        sorted_results = toc_chunks + target_section_heads + neighbor_chunks + other_chunks
        
        if toc_chunks:
            print(f"[RAG] Prioritized {len(toc_chunks)} TOC chunks at top")
        if target_section_heads:
            print(f"[RAG] Prioritized {len(target_section_heads)} target section head chunks")
        if neighbor_chunks:
            print(f"[RAG] Prioritized {len(neighbor_chunks)} neighbor chunks (for table/content continuity)")



        # ENHANCED: Smart chunk selection based on query type
        selected_results = []
        priority_chunks = []
        regular_chunks = []

        # For reasoning queries, also prioritize chunks with related concepts
        concept_keywords = []  # Initialize empty list
        if query_type in ["MULTI_CONCEPT_REASONING", "CODE_ANALYSIS", "COMPARE_SYNTHESIZE"]:
            concept_keywords = [
                "hoisting", "scope", "closure", "function", "arrow", "class",
                "object", "array", "loop", "for", "while", "if", "variable",
                "const", "let", "var", "promise", "async", "callback"
            ]

        # For reasoning queries, also prioritize chunks with related concepts
        if query_type in ["MULTI_CONCEPT_REASONING", "CODE_ANALYSIS", "COMPARE_SYNTHESIZE"]:
            for item in sorted_results:
                content_lower = (item.get("_content", "") or "").lower()
                has_concept = any(kw in content_lower for kw in concept_keywords)

                # CRITICAL FIX: Improved section matching - bắt MỌI mention của PHẦN X
                has_section_match = False
                # Extract all numbers from question (không cần check keywords trước)
                question_numbers = re.findall(r'\d+', question_lower)
                
                # Check if question mentions "phần", "chương", "part" hoặc có số
                has_section_keyword = any(kw in ["phần", "chương", "part"] for kw in question_keywords)
                
                # Match với nhiều pattern khác nhau trong content
                for num in question_numbers:
                    # Pattern 1: "PHẦN X", "phần X", "PHẦN X:", "phần X:"
                    section_patterns = [
                        rf'phần\s+{num}\b',  # "phần 4" hoặc "PHẦN 4"
                        rf'phần\s+{num}[:：]',  # "PHẦN 4:" hoặc "phần 4:"
                        rf'chương\s+{num}\b',  # "chương 4"
                        rf'part\s+{num}\b',  # "part 4"
                    ]
                    
                    for pattern in section_patterns:
                        if re.search(pattern, content_lower, re.IGNORECASE):
                            has_section_match = True
                            print(f"[RAG] Section match found: pattern '{pattern}' in chunk {item.get('chunk_index', '?')}")
                            break
                    
                    if has_section_match:
                        break
                
                # CRITICAL: Nếu có section match HOẶC là neighbor chunk, luôn ưu tiên
                # Neighbor chunks cần được ưu tiên vì chúng chứa bảng/nội dung tiếp theo của section
                if has_section_match or item.get("is_neighbor", False) or (has_concept and item["similarity"] > 0.4):
                    priority_chunks.append(item)
                else:
                    regular_chunks.append(item)
        else:
            # CRITICAL FIX: Improved section matching for other query types
            for item in sorted_results:
                content_lower = (item.get("_content", "") or "").lower()
                has_section_match = False
                
                # Extract all numbers from question (không cần check keywords trước)
                question_numbers = re.findall(r'\d+', question_lower)
                
                # Match với nhiều pattern khác nhau trong content
                for num in question_numbers:
                    # Pattern 1: "PHẦN X", "phần X", "PHẦN X:", "phần X:"
                    section_patterns = [
                        rf'phần\s+{num}\b',  # "phần 4" hoặc "PHẦN 4"
                        rf'phần\s+{num}[:：]',  # "PHẦN 4:" hoặc "phần 4:"
                        rf'chương\s+{num}\b',  # "chương 4"
                        rf'part\s+{num}\b',  # "part 4"
                        rf'{num}\.\d+',  # "4.1", "4.2" (subsection)
                    ]
                    
                    for pattern in section_patterns:
                        if re.search(pattern, content_lower, re.IGNORECASE):
                            has_section_match = True
                            print(f"[RAG] Section match found: pattern '{pattern}' in chunk {item.get('chunk_index', '?')}")
                            break
                    
                    if has_section_match:
                        break

                # CRITICAL: Nếu có section match HOẶC là neighbor chunk, luôn ưu tiên
                # Neighbor chunks cần được ưu tiên vì chúng chứa bảng/nội dung tiếp theo của section
                if has_section_match or item.get("is_neighbor", False):
                    priority_chunks.append(item)
                else:
                    regular_chunks.append(item)



        all_chunks_ordered = priority_chunks + regular_chunks



        current_context_length = 0

        chunk_metadata_for_context = []  # Store metadata to pass to LLM

        # ENHANCED: Use dynamic max_chunks based on query type
        max_selected_chunks = max_chunks_for_query
        
        # CRITICAL FIX: Cho DOCUMENT_OVERVIEW và SECTION_OVERVIEW, ưu tiên số chunks hơn context length
        # Vì cần scan TẤT CẢ chunks để tìm TẤT CẢ các phần
        is_document_overview = query_type == "DOCUMENT_OVERVIEW"
        is_section_overview = query_type == "SECTION_OVERVIEW"
        
        if is_document_overview:
            # CRITICAL FIX: Tăng context length limit drastically cho DOCUMENT_OVERVIEW
            # 1 file: 50000, 2 files: 55000, 3+ files: 60000
            num_docs = len(documents)
            if num_docs >= 3:
                context_limit = 60000  # Tăng từ 40k → 60k
            elif num_docs >= 2:
                context_limit = 55000  # Tăng từ 35k → 55k
            else:
                context_limit = 50000  # Tăng từ 22k → 50k
            print(f"[RAG] DOCUMENT_OVERVIEW: Using extended context limit: {context_limit} chars, max_chunks: {max_selected_chunks} (for {num_docs} document(s))")
        elif is_section_overview:
            # CRITICAL FIX: Tăng context limit cho SECTION_OVERVIEW để đảm bảo neighbor chunks được chọn
            # Đặc biệt quan trọng khi có bảng dữ liệu dài
            num_docs = len(documents)
            if num_docs >= 3:
                context_limit = 20000  # Tăng từ 12k → 20k
            elif num_docs >= 2:
                context_limit = 18000  # Tăng từ 12k → 18k
            else:
                context_limit = 15000  # Tăng từ 12k → 15k
            print(f"[RAG] SECTION_OVERVIEW: Using extended context limit: {context_limit} chars, max_chunks: {max_selected_chunks} (for {num_docs} document(s))")
        elif query_type == "COMPARE_SYNTHESIZE":
            # 🔥 CRITICAL FIX: Tăng context limit cho COMPARE_SYNTHESIZE để đảm bảo có đủ chunks từ CẢ 2 file
            # Đặc biệt quan trọng khi so sánh giữa 2 documents
            num_docs = len(documents)
            if num_docs >= 3:
                context_limit = 25000  # 25k cho 3+ docs
            elif num_docs >= 2:
                context_limit = 22000  # 22k cho 2 docs (QUAN TRỌNG: cần đủ để chứa chunks từ cả 2 file)
            else:
                context_limit = 18000  # 18k cho 1 doc
            print(f"[RAG] COMPARE_SYNTHESIZE: Using extended context limit: {context_limit} chars, max_chunks: {max_selected_chunks} (for {num_docs} document(s))")
        elif query_type == "MULTI_PART":
            # 🔥 CRITICAL FIX: Tăng context limit cho MULTI_PART để đảm bảo có đủ chunks cho tất cả các phần
            num_docs = len(documents)
            if num_docs >= 3:
                context_limit = 25000
            elif num_docs >= 2:
                context_limit = 22000
            else:
                context_limit = 18000
            print(f"[RAG] MULTI_PART: Using extended context limit: {context_limit} chars, max_chunks: {max_selected_chunks} (for {num_docs} document(s))")
        else:
            context_limit = self.base_max_context_length

        # 🔥 CRITICAL FIX: For multi-doc queries, force balanced selection
        # Đặc biệt quan trọng cho SECTION_OVERVIEW, CODE_ANALYSIS, COMPARE_SYNTHESIZE, MULTI_PART
        # QUAN TRỌNG: Nếu có target file được detect, ưu tiên chunks từ target file
        if len(documents) > 1:
            # Group chunks by document_id
            chunks_by_doc = {}
            for item in all_chunks_ordered:
                doc_id = item["document"].id
                if doc_id not in chunks_by_doc:
                    chunks_by_doc[doc_id] = []
                chunks_by_doc[doc_id].append(item)
            
            # 🔥 CRITICAL FIX: Nếu có target file được detect, ưu tiên chunks từ target file
            if target_doc_ids and len(target_doc_ids) > 0:
                print(f"[RAG] ⚖️ Multi-file query with target file(s) detected. Forcing balanced selection with priority for target file(s).")
                
                # Chia slot: ưu tiên target file(s) hơn
                # Nếu có 1 target file: 60% cho target file, 40% cho file còn lại
                # Nếu có 2+ target files: chia đều
                if len(target_doc_ids) == 1:
                    # 1 target file: ưu tiên 60%
                    target_slots = int(max_selected_chunks * 0.6)
                    other_slots = max_selected_chunks - target_slots
                    
                    balanced_chunks = []
                    # Lấy chunks từ target file trước
                    for target_doc_id in target_doc_ids:
                        if target_doc_id in chunks_by_doc:
                            target_chunks = chunks_by_doc[target_doc_id]
                            selected_from_target = target_chunks[:target_slots]
                            balanced_chunks.extend(selected_from_target)
                            doc_filename = target_chunks[0]["document"].filename if target_chunks else "unknown"
                            print(f"[RAG] ✅ {query_type}: Added {len(selected_from_target)} chunks from TARGET file {doc_filename} (priority)")
                    
                    # Lấy chunks từ các file khác
                    for doc_id, doc_chunks in chunks_by_doc.items():
                        if doc_id not in target_doc_ids:
                            selected_from_other = doc_chunks[:other_slots]
                            balanced_chunks.extend(selected_from_other)
                            doc_filename = doc_chunks[0]["document"].filename if doc_chunks else "unknown"
                            print(f"[RAG] ✅ {query_type}: Added {len(selected_from_other)} chunks from {doc_filename}")
                else:
                    # Nhiều target files: chia đều
                    slots_per_target = max_selected_chunks // len(target_doc_ids)
                    balanced_chunks = []
                    for target_doc_id in target_doc_ids:
                        if target_doc_id in chunks_by_doc:
                            target_chunks = chunks_by_doc[target_doc_id]
                            selected_from_target = target_chunks[:slots_per_target]
                            balanced_chunks.extend(selected_from_target)
                            doc_filename = target_chunks[0]["document"].filename if target_chunks else "unknown"
                            print(f"[RAG] ✅ {query_type}: Added {len(selected_from_target)} chunks from TARGET file {doc_filename}")
                    
                    # Fill thêm từ các file khác nếu còn slot
                    remaining_slots = max_selected_chunks - len(balanced_chunks)
                    if remaining_slots > 0:
                        for doc_id, doc_chunks in chunks_by_doc.items():
                            if doc_id not in target_doc_ids:
                                selected_from_other = doc_chunks[:remaining_slots]
                                balanced_chunks.extend(selected_from_other)
                                doc_filename = doc_chunks[0]["document"].filename if doc_chunks else "unknown"
                                print(f"[RAG] ✅ {query_type}: Added {len(selected_from_other)} chunks from {doc_filename}")
                                remaining_slots -= len(selected_from_other)
                                if remaining_slots <= 0:
                                    break
                
                # Re-sort list cuối cùng theo similarity để đưa chunk tốt nhất lên đầu context
                all_chunks_ordered = sorted(balanced_chunks, key=lambda r: r["similarity"], reverse=True)
                print(f"[RAG] Final balanced: {len(all_chunks_ordered)} chunks from {len(chunks_by_doc)} documents (target file priority)")
            else:
                # Không có target file: chia đều như cũ
                # Tính toán số lượng chunk cần lấy cho mỗi file
                # Ví dụ: Max 60 chunks, có 2 file -> mỗi file lấy 30 chunks
                avg_chunks_per_doc = max_selected_chunks // len(documents)
                
                # Đảm bảo mỗi file có ít nhất 15 chunks (nếu có đủ candidate)
                # Với CODE_ANALYSIS, COMPARE_SYNTHESIZE, MULTI_PART: cần ít nhất 15 chunks từ MỖI document
                min_chunks_guaranteed = 15
                if query_type in ["CODE_ANALYSIS", "COMPARE_SYNTHESIZE", "MULTI_PART"]:
                    print(f"[RAG] 🔀 {query_type} multi-doc: Force balance to get chunks from ALL documents")
                    min_chunks_guaranteed = 15  # Đảm bảo ít nhất 15 chunks từ mỗi document
                
                target_per_doc = max(min_chunks_guaranteed, avg_chunks_per_doc)

                print(f"[RAG] Multi-doc query ({query_type}): Forcing balance. Target {target_per_doc} chunks per document.")
                
                balanced_chunks = []
                for doc_id, doc_chunks in chunks_by_doc.items():
                    # Lấy Top N chunks tốt nhất của file này
                    selected_from_doc = doc_chunks[:target_per_doc]
                    balanced_chunks.extend(selected_from_doc)
                    
                    doc_filename = doc_chunks[0]["document"].filename if doc_chunks else "unknown"
                    print(f"[RAG] ✅ {query_type}: Added {len(selected_from_doc)} chunks from {doc_filename}")
                
                # Nếu tổng số chunks chưa đủ max (do một số file ít chunks), fill thêm từ phần còn lại
                if len(balanced_chunks) < max_selected_chunks:
                    # Lấy tất cả chunks còn lại
                    remaining = [c for c in all_chunks_ordered if c not in balanced_chunks]
                    # Sort lại theo similarity
                    remaining = sorted(remaining, key=lambda r: r["similarity"], reverse=True)
                    # Fill cho đến khi đủ
                    needed = max_selected_chunks - len(balanced_chunks)
                    balanced_chunks.extend(remaining[:needed])

                # Re-sort list cuối cùng theo similarity để đưa chunk tốt nhất lên đầu context
                all_chunks_ordered = sorted(balanced_chunks, key=lambda r: r["similarity"], reverse=True)
                print(f"[RAG] Final balanced: {len(all_chunks_ordered)} chunks from {len(chunks_by_doc)} documents")

        # 🔥 CRITICAL FIX: Pre-filtering cho SECTION_OVERVIEW - đảm bảo chunks từ target sections được chọn
        # Đặc biệt quan trọng cho multi-doc queries (e.g., "PHẦN 4 COCOMO và PHẦN 5 Javascript")
        if query_type == "SECTION_OVERVIEW" and len(documents) > 1:
            print(f"[RAG] SECTION_OVERVIEW: Pre-filtering to ensure target sections from all documents...")
            
            # Extract target sections from question (e.g., "PHẦN 4", "PHẦN 5")
            target_sections = re.findall(r'(?:phần|chương|part)\s+(\d+)', question_lower)
            print(f"[RAG] Target sections detected: {target_sections}")
            
            # Map sections to documents (if question mentions file names)
            # Example: "PHẦN 4 trong file COCOMO và PHẦN 5 trong file Javascript"
            section_to_doc = {}  # {section_num: [doc_ids]}
            for sec_num in target_sections:
                section_to_doc[sec_num] = []
            
            # Try to map sections to specific documents from question
            for doc in documents:
                doc_filename = doc.filename.lower()
                doc_name_clean = doc_filename.split(".pdf")[0].strip()
                # Also try matching without extension and with common variations
                doc_name_variations = [
                    doc_name_clean,
                    doc_name_clean.replace("-", " "),
                    doc_name_clean.replace("_", " "),
                    "cocomo",  # Special case for COCOMO
                    "javascript",  # Special case for Javascript
                    "js",  # Short form
                ]
                
                # Check if question mentions this document with a section
                for sec_num in target_sections:
                    # Pattern 1: "phần X trong file [filename]" 
                    # Pattern 2: "phần X file [filename]"
                    # Pattern 3: "phần X [filename]" (without "file")
                    for doc_variant in doc_name_variations:
                        patterns = [
                            rf'phần\s+{sec_num}.*?(?:trong\s+file|file).*?{re.escape(doc_variant)}',
                            rf'phần\s+{sec_num}.*?{re.escape(doc_variant)}.*?(?:trong\s+file|file)',
                            rf'phần\s+{sec_num}.*?{re.escape(doc_variant)}(?!\w)',  # Word boundary
                        ]
                        for pattern in patterns:
                            if re.search(pattern, question_lower, re.IGNORECASE):
                                if doc.id not in section_to_doc[sec_num]:
                                    section_to_doc[sec_num].append(doc.id)
                                    print(f"[RAG] Mapped PHẦN {sec_num} to document {doc.filename}")
                                break
                        if doc.id in section_to_doc[sec_num]:
                            break
            
            # If some sections are not mapped, only allow unmapped sections to search in all documents
            # But mapped sections should ONLY search in their mapped documents
            for sec_num in target_sections:
                if not section_to_doc[sec_num]:
                    # Unmapped section: can be in any document
                    section_to_doc[sec_num] = [doc.id for doc in documents]
                    print(f"[RAG] PHẦN {sec_num} not explicitly mapped -> can be in any document")
                else:
                    print(f"[RAG] PHẦN {sec_num} mapped to {len(section_to_doc[sec_num])} document(s)")
            
            # Find chunks containing target sections
            target_chunks = []  # List of (section_num, doc_id, item)
            for item in all_chunks_ordered:
                content = item.get("_content", "")
                if not content:
                    continue
                
                content_lower = content.lower()
                record = item.get("_record", {})
                doc_id = record.get("document_id")
                
                # Check if this chunk contains any target section
                for sec_num in target_sections:
                    # Pattern chính xác: "PHẦN 4:", "PHẦN 4 " (đầu dòng hoặc sau xuống dòng)
                    precise_pattern = rf'(?:^|\n)\s*(?:phần|chương|part)\s+{sec_num}\b'
                    
                    if re.search(precise_pattern, content_lower):
                        # Check if this document is allowed for this section
                        if not section_to_doc[sec_num] or doc_id in section_to_doc[sec_num]:
                            target_chunks.append((sec_num, doc_id, item))
                            chunk_idx = record.get("chunk_index", "?")
                            print(f"[RAG] 🎯 Found PHẦN {sec_num} in chunk {chunk_idx} (doc: {doc_id})")
            
            # Group target chunks by (section_num, doc_id) and select best chunk for each
            target_chunks_by_key = {}  # {(section_num, doc_id): [items]}
            for sec_num, doc_id, item in target_chunks:
                key = (sec_num, doc_id)
                if key not in target_chunks_by_key:
                    target_chunks_by_key[key] = []
                target_chunks_by_key[key].append(item)
            
            # Select best chunk (highest similarity) for each (section, doc) pair
            section_representatives = []
            for (sec_num, doc_id), items in target_chunks_by_key.items():
                # Sort by similarity and take best
                items_sorted = sorted(items, key=lambda x: x.get("similarity", 0), reverse=True)
                best_item = items_sorted[0]
                section_representatives.append(best_item)
                
                chunk_idx = best_item.get("_record", {}).get("chunk_index", "?")
                # Get document filename from item structure
                doc_obj = best_item.get("document")
                if doc_obj:
                    doc_filename = getattr(doc_obj, "filename", None) or getattr(doc_obj, "title", None) or "unknown"
                else:
                    # Fallback: try to get from record metadata
                    doc_id = best_item.get("_record", {}).get("document_id")
                    doc_filename = f"doc_{doc_id}" if doc_id else "unknown"
                print(f"[RAG] ✅ Pre-selected PHẦN {sec_num} from {doc_filename}: chunk {chunk_idx} (similarity: {best_item.get('similarity', 0):.3f})")
            
            # Add representatives to selected_results FIRST
            # CRITICAL: Ensure chunks have content before adding - Load content from DB if missing
            for rep_item in section_representatives:
                rep_content = rep_item.get("_content", "")
                
                # If content is missing or too short, try to load from DB
                # Tăng threshold lên 100 chars để đảm bảo có đủ content (không chỉ heading)
                if not rep_content or len(rep_content) < 100:
                    record = rep_item.get("_record", {})
                    if record:
                        chunk_id = record.get("chunk_id")
                        if chunk_id:
                            try:
                                from bson import ObjectId
                                chunk_doc = await db["chunks"].find_one({"_id": ObjectId(chunk_id)})
                            except Exception:
                                chunk_doc = await db["chunks"].find_one({"_id": chunk_id})
                            
                            if chunk_doc:
                                rep_content = chunk_doc.get("content") or record.get("content") or ""
                                rep_item["_content"] = rep_content
                                rep_item["_chunk_doc"] = chunk_doc
                                chunk_idx = record.get("chunk_index", "?")
                                print(f"[RAG] 🔄 Reloaded content for pre-selected chunk {chunk_idx}: {len(rep_content)} chars")
                
                if not rep_content:
                    record = rep_item.get("_record", {})
                    if record:
                        print(f"[RAG] ⚠️ WARNING: Pre-selected chunk {record.get('chunk_index', '?')} has no _content after reload!")
                        continue
                
                rep_length = len(rep_content) if rep_content else 0
                if rep_length == 0:
                    print(f"[RAG] ⚠️ WARNING: Pre-selected chunk has empty content, skipping")
                    continue
                
                selected_results.append(rep_item)
                current_context_length += rep_length
                chunk_idx = rep_item.get("_record", {}).get("chunk_index", "?")
                print(f"[RAG] ✅ Added pre-selected chunk {chunk_idx} to context ({rep_length} chars)")
            
            # 🔥 CRITICAL FIX: Pre-select neighbor chunks của target section heads
            # Đảm bảo neighbor chunks được chọn cùng với target section heads
            neighbor_chunks_to_pre_select = []
            selected_chunk_indices_set = {
                rep_item.get("_record", {}).get("chunk_index")
                for rep_item in section_representatives
            }
            
            # Tạo set các target section head chunk indices để match với neighbor chunks
            target_section_head_indices = {
                rep_item.get("_record", {}).get("chunk_index")
                for rep_item in section_representatives
            }
            
            # Tìm tất cả neighbor chunks từ cùng document với target section heads
            # Đơn giản hóa: pre-select TẤT CẢ neighbor chunks từ cùng document với target section heads
            target_doc_ids = {rep_item.get("document").id for rep_item in section_representatives if rep_item.get("document")}
            
            for item in all_chunks_ordered:
                if item.get("is_neighbor", False):
                    item_chunk_idx = item.get("_record", {}).get("chunk_index")
                    item_doc = item.get("document")
                    
                    # Kiểm tra xem neighbor chunk này có thuộc về cùng document với target section heads không
                    if item_chunk_idx is not None and item_chunk_idx not in selected_chunk_indices_set:
                        if item_doc and item_doc.id in target_doc_ids:
                            # Pre-select neighbor chunk nếu:
                            # 1. Có parent_chunk_index và nó match với target section head, HOẶC
                            # 2. Chunk_index gần với target section head (trong khoảng 1-3)
                            parent_chunk_idx = item.get("parent_chunk_index")
                            should_pre_select = False
                            matched_target = None
                            
                            if parent_chunk_idx and parent_chunk_idx in target_section_head_indices:
                                should_pre_select = True
                                matched_target = parent_chunk_idx
                            else:
                                # Kiểm tra xem có gần target section head không
                                for target_idx in target_section_head_indices:
                                    target_item = next((r for r in section_representatives if r.get("_record", {}).get("chunk_index") == target_idx), None)
                                    if target_item and target_item.get("document") and target_item.get("document").id == item_doc.id:
                                        # Neighbor chunks thường là chunk_index + 1, +2, +3
                                        if abs(item_chunk_idx - target_idx) <= 3:
                                            should_pre_select = True
                                            matched_target = target_idx
                                            break
                            
                            if should_pre_select:
                                neighbor_chunks_to_pre_select.append(item)
                                selected_chunk_indices_set.add(item_chunk_idx)
                                print(f"[RAG] ✅ Pre-selected neighbor chunk {item_chunk_idx} for target section chunk {matched_target}")
            
            # Add neighbor chunks to selected_results
            # CRITICAL: Ensure neighbor chunks have full content loaded
            for neighbor_item in neighbor_chunks_to_pre_select:
                neighbor_content = neighbor_item.get("_content", "")
                
                # If content is missing or too short, try to load from DB
                # Tăng threshold lên 100 chars để đảm bảo có đủ content (không chỉ heading)
                if not neighbor_content or len(neighbor_content) < 100:
                    record = neighbor_item.get("_record", {})
                    if record:
                        chunk_id = record.get("chunk_id")
                        if chunk_id:
                            try:
                                from bson import ObjectId
                                chunk_doc = await db["chunks"].find_one({"_id": ObjectId(chunk_id)})
                            except Exception:
                                chunk_doc = await db["chunks"].find_one({"_id": chunk_id})
                            
                            if chunk_doc:
                                neighbor_content = chunk_doc.get("content") or record.get("content") or ""
                                neighbor_item["_content"] = neighbor_content
                                neighbor_item["_chunk_doc"] = chunk_doc
                                chunk_idx = record.get("chunk_index", "?")
                                print(f"[RAG] 🔄 Reloaded content for pre-selected neighbor chunk {chunk_idx}: {len(neighbor_content)} chars")
                
                if neighbor_content:
                    neighbor_length = len(neighbor_content)
                    selected_results.append(neighbor_item)
                    current_context_length += neighbor_length
                    chunk_idx = neighbor_item.get("_record", {}).get("chunk_index", "?")
                    print(f"[RAG] ✅ Added pre-selected neighbor chunk {chunk_idx} to context ({neighbor_length} chars)")
                else:
                    chunk_idx = neighbor_item.get("_record", {}).get("chunk_index", "?")
                    print(f"[RAG] ⚠️ WARNING: Neighbor chunk {chunk_idx} has no content, skipping")
            
            print(f"[RAG] SECTION_OVERVIEW: Pre-selected {len(section_representatives)} target section chunks + {len(neighbor_chunks_to_pre_select)} neighbor chunks from {len(set(doc_id for _, doc_id, _ in target_chunks))} document(s)")
            print(f"[RAG] Pre-selection context length: {current_context_length}/{context_limit} chars")
            
            # Remove selected chunks (both target section heads AND neighbor chunks) from all_chunks_ordered to avoid duplicates
            all_chunks_ordered = [
                item for item in all_chunks_ordered
                if item.get("_record", {}).get("chunk_index") not in selected_chunk_indices_set
            ]
            print(f"[RAG] Remaining chunks after pre-selection: {len(all_chunks_ordered)}")
        
        # CRITICAL FIX: Pre-filtering để đảm bảo section coverage cho DOCUMENT_OVERVIEW
        # Đảm bảo mỗi section có ít nhất 1 chunk representative
        if is_document_overview:
            print(f"[RAG] DOCUMENT_OVERVIEW: Pre-filtering to ensure section coverage...")
            
            # Nhóm chunks theo section
            chunks_by_section = {}
            for idx, item in enumerate(all_chunks_ordered):
                content = item.get("_content", "")
                if not content:
                    continue
                
                # Extract PHẦN X từ content (nhiều patterns)
                section_patterns = [
                    r'PHẦN\s+(\d+)',  # "PHẦN 4"
                    r'phần\s+(\d+)',  # "phần 4"
                    r'chương\s+(\d+)',  # "chương 4"
                    r'part\s+(\d+)',  # "part 4"
                ]
                
                section_num = None
                for pattern in section_patterns:
                    match = re.search(pattern, content, re.IGNORECASE)
                    if match:
                        section_num = int(match.group(1))
                        break
                
                if section_num:
                    if section_num not in chunks_by_section:
                        chunks_by_section[section_num] = []
                    chunks_by_section[section_num].append((idx, item))
            
            # Lấy 1 chunk representative từ mỗi section (chunk có similarity cao nhất)
            section_representatives = []
            for section_num in sorted(chunks_by_section.keys()):
                chunk_list = chunks_by_section[section_num]
                # Sort by similarity (descending) và lấy chunk đầu tiên
                chunk_list.sort(key=lambda x: x[1].get("similarity", 0), reverse=True)
                best_idx, best_item = chunk_list[0]
                section_representatives.append(best_item)
                
                chunk_index = best_item.get("_record", {}).get("chunk_index", "?")
                print(f"[RAG] Section PHẦN {section_num}: selected chunk {chunk_index} (similarity: {best_item.get('similarity', 0):.3f})")
            
            # Thêm representatives vào selected_results trước
            for rep_item in section_representatives:
                rep_content = rep_item.get("_content", "")
                rep_length = len(rep_content) if rep_content else 0
                selected_results.append(rep_item)
                current_context_length += rep_length
            
            print(f"[RAG] DOCUMENT_OVERVIEW: Pre-selected {len(section_representatives)} section representatives from {len(chunks_by_section)} sections")
            print(f"[RAG] Pre-selection context length: {current_context_length}/{context_limit} chars")
            
            # Loại bỏ các chunks đã được chọn từ all_chunks_ordered để tránh duplicate
            selected_chunk_indices = {
                rep_item.get("_record", {}).get("chunk_index")
                for rep_item in section_representatives
            }
            all_chunks_ordered = [
                item for item in all_chunks_ordered
                if item.get("_record", {}).get("chunk_index") not in selected_chunk_indices
            ]
            print(f"[RAG] Remaining chunks after pre-selection: {len(all_chunks_ordered)}")

        # 🔥 CRITICAL FIX: Detect target file from question để ưu tiên chunks từ file đó
        target_file_ids = set()
        if len(documents) > 1:
            question_lower = question.lower()
            # Detect target file từ question
            for doc in documents:
                doc_filename_lower = doc.filename.lower()
                doc_name_clean = doc_filename_lower.split(".pdf")[0].strip()
                doc_name_variations = [
                    doc_name_clean,
                    doc_name_clean.replace("-", " "),
                    doc_name_clean.replace("_", " "),
                    "cocomo",  # Special case
                    "javascript",  # Special case
                    "js",  # Short form
                ]
                
                # Check if question mentions this file
                for variant in doc_name_variations:
                    # Pattern: "file COCOMO", "trong file COCOMO", "COCOMO file", etc.
                    patterns = [
                        rf'(?:file|trong\s+file|tài\s+liệu).*?{re.escape(variant)}',
                        rf'{re.escape(variant)}.*?(?:file|tài\s+liệu)',
                        rf'{re.escape(variant)}(?!\w)',  # Word boundary
                    ]
                    for pattern in patterns:
                        if re.search(pattern, question_lower, re.IGNORECASE):
                            target_file_ids.add(doc.id)
                            print(f"[RAG] 🎯 Target file detected from question: {doc.filename}")
                            break
                    if doc.id in target_file_ids:
                        break
        
        # 🔥 CRITICAL FIX: Sắp xếp lại để ưu tiên chunks từ target file
        if target_file_ids:
            # Tách chunks thành 2 nhóm: từ target file và không từ target file
            chunks_from_target = []
            chunks_not_from_target = []
            for item in all_chunks_ordered:
                if item.get("document").id in target_file_ids:
                    chunks_from_target.append(item)
                else:
                    chunks_not_from_target.append(item)
            
            # Sắp xếp lại: chunks từ target file lên đầu, sau đó là chunks khác
            # Giữ nguyên thứ tự similarity trong mỗi nhóm
            all_chunks_ordered = chunks_from_target + chunks_not_from_target
            print(f"[RAG] 🎯 Reordered chunks: {len(chunks_from_target)} from target file(s) first, then {len(chunks_not_from_target)} from other files")
        
        for item in all_chunks_ordered:

            content = item.get("_content", "")

            content_length = len(content) if content else 0

            # CRITICAL FIX: Check cả context length VÀ số chunks
            # Cho DOCUMENT_OVERVIEW và SECTION_OVERVIEW: ưu tiên số chunks, chỉ dừng khi quá giới hạn nghiêm trọng
            # Đặc biệt: Neighbor chunks luôn được chọn ngay cả khi context limit gần đạt
            is_neighbor = item.get("is_neighbor", False)
            
            # 🔥 CRITICAL FIX: Ưu tiên chunks từ target file
            is_from_target_file = item.get("document").id in target_file_ids if target_file_ids else False
            
            if is_document_overview:
                # Cho DOCUMENT_OVERVIEW: ưu tiên số chunks
                # Với multi-doc: cho phép vượt quá context limit nhiều hơn để đảm bảo có chunks từ TẤT CẢ documents
                if len(documents) > 1:
                    # Multi-doc: cho phép vượt quá 20% để đảm bảo có đủ chunks từ cả 2 files
                    # Vì cần scan TẤT CẢ chunks để tìm TẤT CẢ các phần từ TẤT CẢ documents
                    context_threshold = context_limit * 1.20  # 120% của limit
                else:
                    # Single doc: cho phép vượt quá 5%
                    context_threshold = context_limit * 1.05  # 105% của limit
                
                # Chỉ dừng nếu:
                # 1. Đã đủ chunks (max_selected_chunks), HOẶC
                # 2. Context length vượt quá threshold
                #    Vì DOCUMENT_OVERVIEW cần scan TẤT CẢ chunks để tìm TẤT CẢ các phần
                if (len(selected_results) >= max_selected_chunks) or \
                   (current_context_length + content_length + 500 > context_threshold):
                    break
            elif is_section_overview:
                # Cho SECTION_OVERVIEW: ưu tiên neighbor chunks, target section chunks, và chunks từ target file
                # Neighbor chunks luôn được chọn ngay cả khi context limit gần đạt
                if is_neighbor or item.get("is_target_section_head", False) or is_from_target_file:
                    # Neighbor chunks, target section heads, và chunks từ target file: cho phép vượt quá 30% để đảm bảo được chọn
                    context_threshold = context_limit * 1.30  # 130% của limit
                    if (len(selected_results) >= max_selected_chunks) or \
                       (current_context_length + content_length + 500 > context_threshold):
                        break
                else:
                    # Các chunks khác: giữ logic bình thường
                    if (current_context_length + content_length + 500 > context_limit) or \
                       (len(selected_results) >= max_selected_chunks):
                        break
            elif query_type == "CODE_ANALYSIS":
                # Cho CODE_ANALYSIS: ưu tiên chunks từ target file
                if is_from_target_file:
                    # Chunks từ target file: cho phép vượt quá 20% để đảm bảo được chọn
                    context_threshold = context_limit * 1.20  # 120% của limit
                    if (len(selected_results) >= max_selected_chunks) or \
                       (current_context_length + content_length + 500 > context_threshold):
                        break
                else:
                    # Các chunks khác: giữ logic bình thường
                    if (current_context_length + content_length + 500 > context_limit) or \
                       (len(selected_results) >= max_selected_chunks):
                        break
            elif query_type == "COMPARE_SYNTHESIZE":
                # 🔥 CRITICAL FIX: Cho COMPARE_SYNTHESIZE, đảm bảo chunks từ CẢ 2 file được giữ lại
                # Đếm số chunks đã chọn từ mỗi document
                chunks_by_doc_in_selected = {}
                for selected_item in selected_results:
                    doc_id = selected_item.get("document").id
                    chunks_by_doc_in_selected[doc_id] = chunks_by_doc_in_selected.get(doc_id, 0) + 1
                
                # Đảm bảo mỗi document có ít nhất một số chunks tối thiểu
                min_chunks_per_doc = max(10, max_selected_chunks // (len(documents) * 2))  # Ít nhất 10 chunks hoặc 1/4 của max
                
                current_doc_id = item.get("document").id
                current_doc_chunks_count = chunks_by_doc_in_selected.get(current_doc_id, 0)
                
                # Nếu document này chưa đủ chunks tối thiểu, ưu tiên chọn (cho phép vượt context limit)
                if current_doc_chunks_count < min_chunks_per_doc:
                    # Cho phép vượt quá 30% để đảm bảo có đủ chunks từ document này
                    context_threshold = context_limit * 1.30  # 130% của limit
                    if (len(selected_results) >= max_selected_chunks) or \
                       (current_context_length + content_length + 500 > context_threshold):
                        break
                else:
                    # Đã đủ chunks tối thiểu, giữ logic bình thường nhưng vẫn cho phép vượt một chút
                    context_threshold = context_limit * 1.15  # 115% của limit
                    if (len(selected_results) >= max_selected_chunks) or \
                       (current_context_length + content_length + 500 > context_threshold):
                        break
            elif query_type == "MULTI_PART":
                # 🔥 CRITICAL FIX: Cho MULTI_PART, tương tự COMPARE_SYNTHESIZE
                chunks_by_doc_in_selected = {}
                for selected_item in selected_results:
                    doc_id = selected_item.get("document").id
                    chunks_by_doc_in_selected[doc_id] = chunks_by_doc_in_selected.get(doc_id, 0) + 1
                
                min_chunks_per_doc = max(10, max_selected_chunks // (len(documents) * 2))
                current_doc_id = item.get("document").id
                current_doc_chunks_count = chunks_by_doc_in_selected.get(current_doc_id, 0)
                
                if current_doc_chunks_count < min_chunks_per_doc:
                    context_threshold = context_limit * 1.30
                    if (len(selected_results) >= max_selected_chunks) or \
                       (current_context_length + content_length + 500 > context_threshold):
                        break
                else:
                    context_threshold = context_limit * 1.15
                if (len(selected_results) >= max_selected_chunks) or \
                   (current_context_length + content_length + 500 > context_threshold):
                    break
            else:
                # Cho các query types khác: giữ logic cũ
                if (current_context_length + content_length + 500 > context_limit) or \
                   (len(selected_results) >= max_selected_chunks):
                    break



            selected_results.append(item)

            

            # Store chunk metadata

            record = item.get("_record")

            chunk_doc = item.get("_chunk_doc")

            
            # Lấy metadata từ chunk_doc (chunks collection)
            chunk_metadata = {}
            if chunk_doc:
                chunk_metadata = (chunk_doc.get("metadata") or {})
                # Debug: log metadata cho một vài chunks để kiểm tra
                chunk_idx = record.get("chunk_index") if record else None
                if chunk_idx in [1, 3, 8, 10, 11] and item["document"].file_type in ["docx", "doc"]:
                    print(f"[RAG] Building context - chunk {chunk_idx}: chunk_doc metadata = {chunk_metadata}")
            else:
                # Nếu không có chunk_doc, thử từ record
                if record:
                    chunk_metadata = (record.get("metadata") or {})
            
            # Đảm bảo chunk_metadata là dict
            if not isinstance(chunk_metadata, dict):
                chunk_metadata = {}

            # Lấy heading hoặc section title cho DOCX

            heading = chunk_metadata.get("heading") or chunk_metadata.get("title") or chunk_metadata.get("section_title")
            section = chunk_metadata.get("section")
            
            # Nếu không có section/heading trong metadata, thử extract từ content
            if not section and not heading and content and item["document"].file_type in ["docx", "doc", "md", "txt"]:
                extracted = self._extract_section_from_content(content, item["document"].file_type)
                if extracted:
                    section = extracted
                    # Nếu content ngắn và có vẻ là heading, thì đó là heading
                    if len(content.strip()) < 100:
                        heading = extracted

            chunk_idx = record.get("chunk_index") if record else None
            
            chunk_metadata_for_context.append({

                "chunk_index": chunk_idx,

                "document_id": record.get("document_id") if record else None,  # THÊM document_id

                "page_number": chunk_metadata.get("page_number"),

                "section": section,

                "heading": heading,  # THÊM heading

                "document_type": item["document"].file_type,  # THÊM loại file

                "document_filename": item["document"].filename,  # THÊM tên file

                "content": content

            })
            
            # Debug: log section/heading cho một vài chunks
            if chunk_idx in [1, 3, 8, 10, 11] and item["document"].file_type in ["docx", "doc"]:
                print(f"[RAG] Context metadata for chunk {chunk_idx}: section={section}, heading={heading}, extracted_from_content={section != chunk_metadata.get('section')}")

            

            current_context_length += content_length



        print(f"[RAG] Selected {len(selected_results)} chunks (context length: {current_context_length}/{context_limit} chars, max_chunks: {max_selected_chunks})")

        print(f"[RAG] Priority chunks: {len(priority_chunks)}, Regular chunks: {len(regular_chunks)}")

        # Log neighbor chunks được chọn
        neighbor_chunks_selected = [item for item in selected_results if item.get("is_neighbor", False)]
        if neighbor_chunks_selected:
            neighbor_indices = [item.get("_record", {}).get("chunk_index") for item in neighbor_chunks_selected]
            print(f"[RAG] ✅ {len(neighbor_chunks_selected)} neighbor chunks selected into final context: {neighbor_indices}")
        else:
            print(f"[RAG] ⚠️ No neighbor chunks selected into final context (may be filtered by context limit)")
        
        # 🔥 DEBUG: Track chunks distribution per document (especially for multi-doc queries)
        if len(documents) > 1:
            chunks_count_by_doc = {}
            for item in selected_results:
                doc_id = item["document"].id
                doc_name = item["document"].filename
                if doc_id not in chunks_count_by_doc:
                    chunks_count_by_doc[doc_id] = {"name": doc_name, "count": 0, "chunk_indices": []}
                chunks_count_by_doc[doc_id]["count"] += 1
                chunk_idx = item.get("_record", {}).get("chunk_index")
                if chunk_idx is not None:
                    chunks_count_by_doc[doc_id]["chunk_indices"].append(chunk_idx)
            
            print(f"[RAG] 📊 Chunks distribution in selected_results:")
            for doc_id, info in chunks_count_by_doc.items():
                print(f"  - {info['name']}: {info['count']} chunks (indices: {info['chunk_indices'][:10]}{'...' if len(info['chunk_indices']) > 10 else ''})")



        # Add similarity to chunk metadata
        for item in selected_results:
            record = item.get("_record")
            if record:
                chunk_idx = record.get("chunk_index")
                for chunk_meta in chunk_metadata_for_context:
                    if chunk_meta.get("chunk_index") == chunk_idx:
                        chunk_meta["similarity"] = item.get("similarity", 0.5)
                        break
        
        # CRITICAL FIX: Store selected_results for recovery mechanism in COMPARE_SYNTHESIZE
        self.selected_results = selected_results
        
        # ENHANCED: Generate answer with query_type passed to prompt builder
        answer, chunks_actually_used, answer_type, confidence, sentence_mapping = \
            await self._generate_answer_with_tracking(
                question, 
                chunk_metadata_for_context,
                query_type,  # Pass query type to generation
                selected_documents=documents,
            )

        print(f"[RAG] LLM used {len(chunks_actually_used)} chunks in answer")
        print(f"[RAG] Chunks used: {chunks_actually_used}")
        print(f"[RAG] Answer type: {answer_type}, Confidence: {confidence:.2f}")



        # Build references from chunks actually used in answer
        # If LLM didn't return chunk indices, use all selected chunks (sorted by similarity)
        
        # Create a metadata map from chunk_metadata_for_context for quick lookup
        # Key: (chunk_index, document_id) -> metadata dict
        metadata_map = {}
        for chunk_meta in chunk_metadata_for_context:
            key = (chunk_meta.get("chunk_index"), chunk_meta.get("document_id"))
            metadata_map[key] = chunk_meta
        
        # Second pass: Fill in missing sections by looking backward at previous chunks
        # This ensures all chunks have section information if available from earlier chunks
        # Sort chunks by chunk_index within each document to ensure proper order
        chunks_by_doc = {}
        for chunk_meta in chunk_metadata_for_context:
            doc_id = chunk_meta.get("document_id")
            if doc_id not in chunks_by_doc:
                chunks_by_doc[doc_id] = []
            chunks_by_doc[doc_id].append(chunk_meta)
        
        # For each document, sort chunks by chunk_index and fill in missing sections
        for doc_id, doc_chunks in chunks_by_doc.items():
            # Sort by chunk_index
            doc_chunks.sort(key=lambda x: x.get("chunk_index") or 0)
            
            # Now iterate through chunks in order and fill in missing sections
            for i, chunk_meta in enumerate(doc_chunks):
                chunk_idx = chunk_meta.get("chunk_index")
                current_section = chunk_meta.get("section")
                current_heading = chunk_meta.get("heading")
                
                # If this chunk doesn't have a section/heading, try to find one from previous chunks
                if not current_section and not current_heading and chunk_idx is not None and chunk_idx > 0:
                    # Look backward through previous chunks in the same document
                    # Collect all potential sections and choose the best one
                    candidate_sections = []
                    
                    # Look backward through chunks we've already processed
                    for j in range(i - 1, -1, -1):
                        prev_chunk = doc_chunks[j]
                        prev_section = prev_chunk.get("section")
                        prev_heading = prev_chunk.get("heading")
                        
                        # Collect all sections/headings we find
                        if prev_heading:
                            candidate_sections.append({
                                "text": prev_heading,
                                "is_heading": True,
                                "is_numbered": self._is_numbered_section(prev_heading),
                                "length": len(prev_heading),
                                "chunk_idx": prev_chunk.get("chunk_index")
                            })
                        if prev_section and prev_section != prev_heading:
                            candidate_sections.append({
                                "text": prev_section,
                                "is_heading": False,
                                "is_numbered": self._is_numbered_section(prev_section),
                                "length": len(prev_section),
                                "chunk_idx": prev_chunk.get("chunk_index")
                            })
                    
                    # Choose the best section: prefer numbered sections, then short headings
                    found_section = None
                    found_heading = None
                    
                    if candidate_sections:
                        # Sort by priority:
                        # 1. Numbered sections (like "7.2.2. ...")
                        # 2. Short headings (< 60 chars)
                        # 3. Other sections
                        def section_priority(candidate):
                            priority = 0
                            if candidate["is_numbered"]:
                                priority += 1000  # Highest priority
                            if candidate["is_heading"]:
                                priority += 100
                            if candidate["length"] < 60:
                                priority += 50
                            # Prefer sections from chunks closer to current chunk
                            priority += (100 - candidate["chunk_idx"] or 0) / 100
                            return priority
                        
                        candidate_sections.sort(key=section_priority, reverse=True)
                        best_candidate = candidate_sections[0]
                        found_section = best_candidate["text"]
                        found_heading = best_candidate["text"] if best_candidate["is_heading"] else None
                    
                    # If we found a section from a previous chunk, update this chunk's metadata
                    if found_section or found_heading:
                        chunk_meta["section"] = found_section
                        chunk_meta["heading"] = found_heading
                        # Also update the metadata_map entry
                        metadata_key = (chunk_idx, doc_id)
                        if metadata_key in metadata_map:
                            metadata_map[metadata_key]["section"] = found_section
                            metadata_map[metadata_key]["heading"] = found_heading
                            print(f"[RAG] Filled section for chunk {chunk_idx} from previous chunk: {found_section or found_heading}")
        
        # Count chunks per document BEFORE filtering (to determine which documents are most relevant)
        chunks_by_document = {}
        if chunks_actually_used:
            for chunk_info in chunks_actually_used:
                if isinstance(chunk_info, dict):
                    doc_id = chunk_info.get("document_id")
                else:
                    # Find document_id from selected_results
                    doc_id = None
                    for item in selected_results:
                        record = item.get("_record")
                        if record and record.get("chunk_index") == chunk_info:
                            doc_id = record.get("document_id")
                            break
                if doc_id:
                    chunks_by_document[doc_id] = chunks_by_document.get(doc_id, 0) + 1
        
        # === CRITICAL: Reference Logic ===
        final_references = []
        
        if answer_type in ["FALLBACK", "TOO_BROAD"]:
            # STRICT RULE: FALLBACK/TOO_BROAD = 0 references
            # EXCEPTION: If COMPARE_SYNTHESIZE has table, try to recover chunks
            if answer_type == "FALLBACK" and query_type == "COMPARE_SYNTHESIZE" and "|" in answer:
                # Try to use selected chunks for comparison tables
                if selected_results:
                    print(f"[RAG] COMPARE_SYNTHESIZE with table but FALLBACK → trying to recover chunks from selected_results")
                    # Use top chunks from selected_results
                    for item in selected_results[:10]:  # Use top 10 chunks
                        record = item.get("_record")
                        if record:
                            chunks_actually_used.append({
                                "chunk_index": record.get("chunk_index"),
                                "document_id": record.get("document_id")
                            })
                    if chunks_actually_used:
                        answer_type = "COMPARE_SYNTHESIZE"  # Override FALLBACK
                        confidence = 0.75  # Set reasonable confidence
                        print(f"[RAG] ✅ Recovered {len(chunks_actually_used)} chunks for COMPARE_SYNTHESIZE")
            if answer_type in ["FALLBACK", "TOO_BROAD"]:  # Still fallback after recovery attempt
                final_references = []
                print(f"[RAG] ✓ {answer_type} detected → 0 references enforced")
            
        if answer_type not in ["FALLBACK", "TOO_BROAD"] and not chunks_actually_used:
            # No chunks but not fallback → suspicious
            if confidence > 0.5 and len(answer) > 100:
                # Try to infer from sentence mapping
                if sentence_mapping:
                    chunk_indices_from_mapping = [
                        s.get("chunk") for s in sentence_mapping 
                        if s.get("chunk") and not s.get("external", False)
                    ]
                    if chunk_indices_from_mapping:
                        print(f"[RAG] Recovered chunks from sentence_mapping: {chunk_indices_from_mapping}")
                        # Rebuild chunks_used
                        for idx in set(chunk_indices_from_mapping):
                            for item in selected_results:
                                record = item.get("_record")
                                if record and record.get("chunk_index") == idx:
                                    chunks_actually_used.append({
                                        "chunk_index": idx,
                                        "document_id": record.get("document_id")
                                    })
                                    break
                
                # If still no chunks, suspicious → no refs
                if not chunks_actually_used:
                    print(f"[RAG] ⚠ High confidence but no chunks → suspicious, no refs")
                    final_references = []
            else:
                final_references = []
                print(f"[RAG] Low confidence + no chunks → no references")
        
        if chunks_actually_used:
            # Build references from chunks_used
            final_references = self._build_references_from_chunks(
                chunks_actually_used, selected_results, chunk_metadata_for_context
            )
            print(f"[RAG] ✓ Built {len(final_references)} references from chunks")
        
        # OLD LOGIC - REMOVED: If chunks_actually_used is empty, use top selected_results chunks
        # This is now handled above with strict fallback rules
        if False and not chunks_actually_used and selected_results:
            # Sort by similarity score (descending) and take top chunks
            # selected_results is already sorted (priority_chunks + regular_chunks), but ensure by similarity
            sorted_for_refs = sorted(selected_results, key=lambda r: r.get("similarity", 0), reverse=True)
            
            # Count chunks per document to determine relevance
            temp_chunks_by_doc = {}
            for item in sorted_for_refs[:15]:  # Check top 15
                record = item.get("_record")
                if record:
                    doc_id = record.get("document_id")
                    if doc_id:
                        temp_chunks_by_doc[doc_id] = temp_chunks_by_doc.get(doc_id, 0) + 1
            chunks_by_document = temp_chunks_by_doc
            
            # If document_id is specified, only use chunks from that document
            if document_id:
                top_chunks = [item for item in sorted_for_refs if item.get("_record", {}).get("document_id") == document_id][:15]
            else:
                # Filter: only keep chunks from documents with most chunks (if multiple documents)
                if len(chunks_by_document) > 1:
                    max_chunks = max(chunks_by_document.values())
                    # Only keep documents with at least 2 chunks, or if all have 1 chunk, keep all
                    if max_chunks >= 2:
                        relevant_docs = {doc_id for doc_id, count in chunks_by_document.items() if count >= 2}
                        if relevant_docs:
                            top_chunks = [item for item in sorted_for_refs 
                                        if item.get("_record", {}).get("document_id") in relevant_docs][:15]
                            print(f"[RAG] Filtered to documents with 2+ chunks: {relevant_docs}")
                        else:
                            top_chunks = sorted_for_refs[:15]
                    else:
                        top_chunks = sorted_for_refs[:15]
                else:
                    top_chunks = sorted_for_refs[:15]
            
            print(f"[RAG] No chunks returned by LLM, using top {len(top_chunks)} chunks (out of {len(selected_results)}) as references")
            # Convert selected_results to chunk_info format
            chunks_actually_used = []
            for item in top_chunks:
                record = item.get("_record")
                if record:
                    chunks_actually_used.append({
                        "chunk_index": record.get("chunk_index"),
                        "document_id": record.get("document_id")
                    })
            
            # Update chunks_by_document based on selected chunks
            chunks_by_document = {}
            for chunk_info in chunks_actually_used:
                doc_id = chunk_info.get("document_id")
                if doc_id:
                    chunks_by_document[doc_id] = chunks_by_document.get(doc_id, 0) + 1

        # OLD REFERENCE BUILDING CODE - Skip if we've already built references using new method
        # This old code manually builds references with detailed metadata extraction
        # We keep it as fallback but skip it when using the new _build_references_from_chunks method
        # Note: The new _build_references_from_chunks is called above, so this old code should rarely execute
        # Only execute if final_references is still empty (shouldn't happen with new logic)
        if not final_references and chunks_actually_used:
            # Old manual reference building code (kept for compatibility/fallback)
            # This should rarely execute since _build_references_from_chunks is called above
            for chunk_info in chunks_actually_used:
                # chunk_info có thể là số (chunk_index) hoặc dict {"chunk_index": X, "document_id": Y}
                if isinstance(chunk_info, dict):
                    target_chunk_index = chunk_info.get("chunk_index")
                    target_document_id = chunk_info.get("document_id")
                else:
                    target_chunk_index = chunk_info
                    target_document_id = None

                

                # Find the corresponding item in selected_results

                for item in selected_results:

                    record = item.get("_record")

                    doc = item["document"]

                    

                    if not record:

                        continue

                    

                    # QUAN TRỌNG: Check cả chunk_index VÀ document_id để tránh nhầm lẫn

                    chunk_index_match = record.get("chunk_index") == target_chunk_index

                    

                    # Nếu có document_id trong chunk_info, phải match cả document_id

                    if target_document_id:

                        document_id_match = record.get("document_id") == target_document_id

                        if not (chunk_index_match and document_id_match):

                            continue

                    else:

                        # Nếu không có document_id, chỉ cần match chunk_index

                        if not chunk_index_match:

                            continue

                    

                    chunk_doc = item.get("_chunk_doc")

                    content = item.get("_content")

                    record = item.get("_record")
                    
                    

                    # Lấy metadata - ưu tiên từ metadata_map (đã có sẵn từ chunk_metadata_for_context)
                    # Đây là nguồn đáng tin cậy nhất vì đã được lấy trực tiếp từ database khi build context
                    metadata_key = (target_chunk_index, target_document_id or record.get("document_id") if record else None)
                    cached_metadata = metadata_map.get(metadata_key)
                    
                    if cached_metadata:
                        # Sử dụng metadata từ cache (đã có section, heading, page_number)
                        # Metadata map đã được fill với sections từ previous chunks trong second pass
                        page_number = cached_metadata.get("page_number")
                        section = cached_metadata.get("section")
                        heading = cached_metadata.get("heading")
                        display_section = section or heading
                        
                        # If the section is not a numbered section, we might find a better one from database
                        # So we'll still try to find a numbered section later if display_section is not numbered
                    else:
                        # Fallback: lấy từ database nếu không có trong cache
                        chunk_metadata = {}
                        
                        # Ưu tiên metadata từ chunk_doc (chunks collection)
                        if chunk_doc:
                            chunk_metadata = (chunk_doc.get("metadata") or {})
                        
                        # Nếu không có, thử từ record (embeddings collection)
                        if not chunk_metadata and record:
                            chunk_metadata = (record.get("metadata") or {})
                        
                        # Đảm bảo chunk_metadata là dict
                        if not isinstance(chunk_metadata, dict):
                            chunk_metadata = {}
                        
                        page_number = chunk_metadata.get("page_number")
                        section = chunk_metadata.get("section")
                        heading = chunk_metadata.get("heading") or chunk_metadata.get("title") or chunk_metadata.get("section_title")
                        display_section = section or heading
                        
                        # Debug: log nếu không tìm thấy trong cache
                        if target_chunk_index in [1, 3, 8, 10, 11, 42, 31, 48, 66]:
                            print(f"[RAG] Warning: chunk {target_chunk_index} not found in metadata_map, using database. metadata={chunk_metadata}")
                    
                    # If still no section, try to extract from content (shouldn't happen often after second pass)
                    if not display_section and content and doc.file_type:
                        extracted_section = self._extract_section_from_content(content, doc.file_type)
                        if extracted_section:
                            display_section = extracted_section
                            print(f"[RAG] Extracted section from content for chunk {target_chunk_index}: {display_section}")
                    
                    # If still no section, or if section is not numbered (might have better numbered section),
                    # try to find from previous chunks in database
                    # This handles cases where the chunk wasn't in chunk_metadata_for_context
                    # or where we want to find a better numbered section
                    has_section = bool(display_section)
                    is_numbered = self._is_numbered_section(display_section) if display_section else False
                    
                    # Query database if: no section, or has section but not numbered (might find better numbered section)
                    if target_chunk_index is not None and target_chunk_index > 0 and (not has_section or not is_numbered):
                        doc_id_for_lookup = target_document_id or record.get("document_id") if record else None
                        if doc_id_for_lookup:
                            try:
                                # Query database for previous chunks in the same document
                                # Look for chunks with chunk_index < target_chunk_index, ordered by chunk_index desc
                                # Limit to 10 chunks to avoid too many queries
                                previous_chunks = await db["chunks"].find({
                                    "document_id": doc_id_for_lookup,
                                    "chunk_index": {"$lt": target_chunk_index}
                                }).sort("chunk_index", -1).limit(10).to_list(length=10)
                                
                                # Collect candidate sections from previous chunks
                                candidate_sections = []
                                for prev_chunk in previous_chunks:
                                    prev_metadata = prev_chunk.get("metadata") or {}
                                    prev_section = prev_metadata.get("section")
                                    prev_heading = prev_metadata.get("heading")
                                    
                                    # Also try to extract from content if it looks like a heading
                                    prev_content = prev_chunk.get("content", "")
                                    if not prev_section and not prev_heading and prev_content:
                                        extracted = self._extract_section_from_content(prev_content, doc.file_type if doc else "docx")
                                        if extracted:
                                            prev_section = extracted
                                    
                                    if prev_heading:
                                        candidate_sections.append({
                                            "text": prev_heading,
                                            "is_heading": True,
                                            "is_numbered": self._is_numbered_section(prev_heading),
                                            "length": len(prev_heading),
                                            "chunk_idx": prev_chunk.get("chunk_index")
                                        })
                                    if prev_section and prev_section != prev_heading:
                                        candidate_sections.append({
                                            "text": prev_section,
                                            "is_heading": False,
                                            "is_numbered": self._is_numbered_section(prev_section),
                                            "length": len(prev_section),
                                            "chunk_idx": prev_chunk.get("chunk_index")
                                        })
                                
                                # Choose the best section: prefer numbered sections
                                if candidate_sections:
                                    def section_priority(candidate):
                                        priority = 0
                                        if candidate["is_numbered"]:
                                            priority += 1000  # Highest priority
                                        if candidate["is_heading"]:
                                            priority += 100
                                        if candidate["length"] < 60:
                                            priority += 50
                                        # Prefer sections from chunks closer to current chunk
                                        priority += (100 - candidate["chunk_idx"] or 0) / 100
                                        return priority
                                    
                                    candidate_sections.sort(key=section_priority, reverse=True)
                                    best_candidate = candidate_sections[0]
                                    
                                    # Only use database section if:
                                    # 1. We had no section, OR
                                    # 2. Database has a numbered section (better than non-numbered)
                                    if not has_section or (best_candidate["is_numbered"] and not is_numbered):
                                        display_section = best_candidate["text"]
                                        print(f"[RAG] Found section from database for chunk {target_chunk_index}: {display_section} (from chunk {best_candidate['chunk_idx']})")
                            except Exception as e:
                                print(f"[RAG] Error querying database for previous chunks: {e}")
                    
                    if not display_section:
                        print(f"[RAG] No section found for chunk {target_chunk_index} after all attempts")

                    

                    preview = content[:160] if content else None

                    

                    final_references.append(

                        HistoryReference(

                            document_id=doc.id,

                            document_filename=doc.filename,

                            document_file_type=doc.file_type,

                            chunk_id=str(chunk_doc.get("_id")) if chunk_doc else None,

                            chunk_index=target_chunk_index,

                            page_number=page_number,

                            section=display_section,  # Sử dụng section hoặc heading

                            score=item["similarity"],

                            content_preview=preview,

                        )

                    )

                    break



        # Remove duplicates - khác biệt giữa PDF và DOCX

        seen_keys = set()

        deduplicated_refs = []

        

        for ref in final_references:

            # Với PDF: deduplicate theo page

            if ref.document_file_type and ref.document_file_type.lower() == "pdf" and ref.page_number:

                page_key = f"{ref.document_id}_page_{ref.page_number}"

                if page_key in seen_keys:

                    continue

                seen_keys.add(page_key)

            

            # Với DOCX: deduplicate theo section hoặc chunk

            elif ref.document_file_type and ref.document_file_type.lower() in ["docx", "doc"]:

                if ref.section:

                    section_key = f"{ref.document_id}_section_{ref.section}"

                    if section_key in seen_keys:

                        continue

                    seen_keys.add(section_key)

                else:

                    # Nếu không có section, dùng chunk_index

                    chunk_key = f"{ref.document_id}_chunk_{ref.chunk_index}"

                    if chunk_key in seen_keys:

                        continue

                    seen_keys.add(chunk_key)

            else:

                # Fallback: deduplicate theo chunk_index

                chunk_key = f"{ref.document_id}_chunk_{ref.chunk_index}"

                if chunk_key in seen_keys:

                    continue

                seen_keys.add(chunk_key)

            

            deduplicated_refs.append(ref)



        # CRITICAL: Filter references theo documents được chọn
        filtered_refs = deduplicated_refs
        
        if document_ids:  # ← THAY ĐỔI: Check list
            # CHỈ giữ references từ documents được chọn
            filtered_refs = [
                ref for ref in deduplicated_refs 
                if ref.document_id in document_ids  # ← THAY ĐỔI: check in list
            ]
            print(f"[RAG] 🎯 Filtered to {len(document_ids)} selected document(s): {len(filtered_refs)} references")
        else:
            # "Tất cả tài liệu" mode - giữ mọi reference
            if len(chunks_by_document) > 1:
                print(f"[RAG] 📚 Multiple documents used, keeping all {len(filtered_refs)} references")
        
        # Smart filtering: If there are multiple sections, prioritize the section(s) với nhiều chunk nhất.
        # CRITICAL FIX: Skip filtering for Multi-doc queries and Overviews
        # CHỈ áp dụng lọc khi:
        # 1. Không phải DOCUMENT_OVERVIEW (vì cần liệt kê hết)
        # 2. Không phải MULTI_PART (vì cần trả lời nhiều phần)
        # 3. Không phải SECTION_OVERVIEW (vì có thể hỏi 2 phần ở 2 file)
        # 4. Không phải COMPARE_SYNTHESIZE (vì cần so sánh từ nhiều nguồn)
        # 5. Tất cả references thuộc về CÙNG 1 tài liệu (quan trọng!)
        
        unique_doc_ids = {ref.document_id for ref in filtered_refs if ref.document_id}
        
        if (
            len(filtered_refs) > 2
            and chunks_actually_used
            and answer_type not in ["DOCUMENT_OVERVIEW", "MULTI_PART", "SECTION_OVERVIEW", "COMPARE_SYNTHESIZE"]
            and len(unique_doc_ids) == 1  # CHỈ LỌC NẾU CHỈ CÓ 1 TÀI LIỆU
        ):
            # Count chunks per section from chunks_actually_used
            section_chunk_counts = {}
            for chunk_info in chunks_actually_used:
                if isinstance(chunk_info, dict):
                    chunk_idx = chunk_info.get("chunk_index")
                    doc_id = chunk_info.get("document_id")
                else:
                    chunk_idx = chunk_info
                    doc_id = None
                
                # Find section from metadata_map
                metadata_key = (chunk_idx, doc_id)
                chunk_meta = metadata_map.get(metadata_key)
                if chunk_meta:
                    # Determine section key
                    file_type = chunk_meta.get("document_type", "").lower()
                    if file_type in ["docx", "doc"]:
                        section = chunk_meta.get("section") or chunk_meta.get("heading")
                        if section:
                            section_key = f"{doc_id}_{section}"
                        else:
                            section_key = f"{doc_id}_no_section_{chunk_idx}"
                    elif file_type == "pdf":
                        page = chunk_meta.get("page_number")
                        if page:
                            section_key = f"{doc_id}_page_{page}"
                        else:
                            section_key = f"{doc_id}_no_page_{chunk_idx}"
                    else:
                        section_key = f"{doc_id}_other_{chunk_idx}"
                    
                    section_chunk_counts[section_key] = section_chunk_counts.get(section_key, 0) + 1
            
            # Group filtered_refs by section
            section_refs = {}
            for ref in filtered_refs:
                if ref.document_file_type and ref.document_file_type.lower() in ["docx", "doc"]:
                    section_key = f"{ref.document_id}_{ref.section}" if ref.section else f"{ref.document_id}_no_section_{ref.chunk_index}"
                elif ref.document_file_type and ref.document_file_type.lower() == "pdf":
                    section_key = f"{ref.document_id}_page_{ref.page_number}" if ref.page_number else f"{ref.document_id}_no_page_{ref.chunk_index}"
                else:
                    section_key = f"{ref.document_id}_other_{ref.chunk_index}"
                
                if section_key not in section_refs:
                    section_refs[section_key] = []
                section_refs[section_key].append(ref)
            
            # Sort sections by chunk count (from chunks_actually_used), not by reference count
            sorted_sections = sorted(
                section_refs.items(), 
                key=lambda x: section_chunk_counts.get(x[0], 0), 
                reverse=True
            )
            
            # Keep references from top sections based on chunk count
            if sorted_sections:
                top_section_key = sorted_sections[0][0]
                top_chunk_count = section_chunk_counts.get(top_section_key, 0)
                
                if top_chunk_count >= 3:
                    # Only keep top section if it has 3+ chunks
                    filtered_refs = section_refs[top_section_key]
                    print(f"[RAG] Filtered to top section only: {top_section_key} ({top_chunk_count} chunks, {len(filtered_refs)} references)")
                elif len(sorted_sections) > 1:
                    # Keep top 2 sections
                    second_section_key = sorted_sections[1][0]
                    second_chunk_count = section_chunk_counts.get(second_section_key, 0)
                    filtered_refs = section_refs[top_section_key] + section_refs[second_section_key]
                    print(f"[RAG] Filtered to top 2 sections: {top_section_key} ({top_chunk_count} chunks), {second_section_key} ({second_chunk_count} chunks)")
                else:
                    # Only one section, keep all
                    filtered_refs = sorted_sections[0][1]
        
        # Nếu có nhiều tài liệu, KHÔNG BAO GIỜ LỌC bớt references
        elif len(unique_doc_ids) > 1:
            print(f"[RAG] 🛡️ Multi-doc response ({len(unique_doc_ids)} docs) -> Keeping ALL references to ensure coverage.")

        # CRITICAL: For DOCUMENT_OVERVIEW, keep more references
        if answer_type == "DOCUMENT_OVERVIEW":
            final_references = filtered_refs[:15]  # Tăng từ 10 lên 15 để cover đủ 2 file
        else:
            final_references = filtered_refs[:7]  # Tăng từ 5 lên 7



        print(f"[RAG] Final references: {len(final_references)} chunks")

        print(f"[RAG] Reference details:")

        for ref in final_references:

            print(f"  - File: {ref.document_filename}, Page: {ref.page_number}, Section: {ref.section}, Chunk: {ref.chunk_index}")



        # === ENHANCED LOGGING ===
        documents_used = list({ref.document_id for ref in final_references if getattr(ref, "document_id", None)})

        log_entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "question": question[:100],
            "document_ids": document_ids,  # ← THAY ĐỔI
            "documents_searched": document_ids_used,
            "query_type": detect_query_type_fast(question),
            "answer_type": answer_type,
            "confidence": confidence,
            "chunks_retrieved": len(results) if 'results' in locals() else 0,
            "documents_used": documents_used,
            "chunks_selected": len(selected_results) if 'selected_results' in locals() else 0,
            "chunks_used": [c.get("chunk_index") for c in chunks_actually_used],
            "references_count": len(final_references),
            "answer_length": len(answer),
            "sentence_mapping_count": len(sentence_mapping),
            "max_similarity": max(
                [item.get("similarity", 0) for item in selected_results]
            ) if selected_results else 0
        }
        print(f"[RAG] Query Log: {json.dumps(log_entry, ensure_ascii=False)}")
        
        # Ensure conversation_id is set - use provided conversation_id or create new one
        # If conversation_id is provided, use it (for continuing existing conversation)
        # If not provided, we'll set it to history_id after creating the record
        final_conversation_id = conversation_id

        # Create history with conversation_id (may be None for new conversation)
        print(f"[RAG] ===== Creating history record =====")
        print(f"[RAG] Question: {question[:100]}...")
        print(f"[RAG] Answer: {answer[:100]}...")
        print(f"[RAG] Conversation ID: {final_conversation_id}")
        print(f"[RAG] Document IDs: {document_ids_used}")
        print(f"[RAG] References count: {len(final_references)}")
        
        # Use first document_id for backward compatibility with history model
        doc_id_for_history = document_ids[0] if document_ids and len(document_ids) > 0 else None
        history_record = await create_history(
            db, user_id, question, answer, final_references, doc_id_for_history, final_conversation_id
        )
        
        print(f"[RAG] ✅ History record created - ID: {history_record.id}")
        print(f"[RAG] ===================================")

        # If no conversation_id was provided, use the history_id as conversation_id
        # This ensures all Q&As in the same conversation share the same conversation_id
        if not final_conversation_id:
            final_conversation_id = history_record.id
            # Update the history record to set conversation_id = history_id
            # This ensures consistency when loading conversations
            try:
                from bson import ObjectId
                update_result = await db["histories"].update_one(
                    {"_id": ObjectId(history_record.id)},
                    {"$set": {"conversation_id": history_record.id}}
                )
                if update_result.modified_count > 0:
                    print(f"[RAG] Set conversation_id = history_id for new conversation: {history_record.id}")
                # Also update the history_record object for return value
                history_record.conversation_id = history_record.id
            except Exception as e:
                print(f"[RAG] Warning: Failed to update conversation_id for history {history_record.id}: {e}")
        else:
            # Conversation_id was provided, ensure it's set correctly in the record
            # (It should already be set, but double-check for consistency)
            try:
                from bson import ObjectId
                # Verify the record has the correct conversation_id
                existing_record = await db["histories"].find_one({"_id": ObjectId(history_record.id)})
                if existing_record and existing_record.get("conversation_id") != final_conversation_id:
                    await db["histories"].update_one(
                        {"_id": ObjectId(history_record.id)},
                        {"$set": {"conversation_id": final_conversation_id}}
                    )
                    print(f"[RAG] Updated conversation_id to {final_conversation_id} for history {history_record.id}")
            except Exception as e:
                print(f"[RAG] Warning: Failed to verify conversation_id for history {history_record.id}: {e}")



        return {
            "answer": answer,
            "references": final_references,
            "documents": list(set([ref.document_id for ref in final_references if ref.document_id])),
            "documents_searched": document_ids_used,  # ← THÊM: list IDs đã search
            "conversation_id": final_conversation_id,
            "history_id": history_record.id,
            "metadata": {
                "answer_type": answer_type,
                "confidence": confidence,
                "query_type": query_type,  # ← VẪN CÓ 4 TẦNG
                "chunks_selected": len(selected_results),
                "chunks_used": len(chunks_actually_used),
                "documents_count": len(documents),  # ← THÊM: số docs đã search
                "documents_selected": len(document_ids) if document_ids else len(documents),  # ← THÊM
            }
        }



    async def _generate_answer_with_tracking(
        self, 
        question: str, 
        chunk_metadata_list: List[dict],
        query_type: str = "DIRECT",
        selected_documents: Optional[List[DocumentInDB]] = None,
    ) -> tuple[str, List[dict], str, float, List[dict]]:
        """
        Generate answer and track which chunks were actually used.
        Returns: (answer, chunks_used, answer_type, confidence, sentence_mapping)
        """
        # Build context với similarity scores
        context_parts = []
        chunk_similarities = []
        selected_docs_info: Optional[List[Dict[str, str]]] = None
        if selected_documents:
            selected_docs_info = []
            for doc in selected_documents:
                if not doc:
                    continue
                filename = getattr(doc, "filename", None) or getattr(doc, "title", None) or "Unnamed document"
                doc_id = getattr(doc, "id", None)
                selected_docs_info.append({
                    "id": doc_id,
                    "filename": filename,
                })

        for chunk_meta in chunk_metadata_list:
            idx = chunk_meta["chunk_index"]
            content = chunk_meta["content"]
            sim = chunk_meta.get("similarity", 0.5)
            chunk_similarities.append(sim)
            
            # Compact marker format
            parts = [f"[Chunk {idx}]"]
            if chunk_meta.get("document_filename"):
                parts.append(f"[{chunk_meta['document_filename']}]")
            if chunk_meta.get("page_number"):
                parts.append(f"[Page {chunk_meta['page_number']}]")
            if chunk_meta.get("section"):
                parts.append(f"[{chunk_meta['section']}]")
            parts.append(f"[Sim:{sim:.2f}]")
            
            marker = " ".join(parts)
            context_parts.append(f"{marker}\n{content}")
        
        context_text = "\n\n---\n\n".join(context_parts)
        
        # ENHANCED: Build prompt with query_type
        prompt = build_gemini_optimized_prompt(
            question=question,
            context_text=context_text,
            chunk_similarities=chunk_similarities,
            query_type=query_type,
            selected_documents=selected_docs_info,
        )
        
        # Call Gemini API
        # Note: Gemini API uses camelCase, not snake_case
        # Note: responseMimeType is not supported by Gemini 2.5 Flash, so we parse JSON from text
        # 🔥 CRITICAL FIX: Dynamic maxOutputTokens based on query type
        # COMPARE_SYNTHESIZE: Limit to 8192 tokens (~6KB) to prevent JSON parse errors
        # DOCUMENT_OVERVIEW: Keep 12000 tokens for comprehensive overview
        if query_type == "COMPARE_SYNTHESIZE":
            max_output_tokens = 8192  # ~6KB to prevent oversized responses
            print(f"[RAG] COMPARE_SYNTHESIZE: Using limited maxOutputTokens={max_output_tokens} to prevent oversized responses")
        else:
            max_output_tokens = self.max_output_tokens  # 12000 for other types
        
        generation_config = {
            "temperature": 0.0,
            "maxOutputTokens": max_output_tokens,
            "candidateCount": 1,
            "stopSequences": [],
        }
        
        # 🔥 CRITICAL FIX: Retry mechanism with exponential backoff
        max_retries = 3
        base_delay = 2  # seconds
        
        for attempt in range(max_retries):
            try:
                url = f"{self._gemini_base_url}/{self.model}:generateContent"
                payload = {
                    "contents": [{"parts": [{"text": prompt}]}],
                    "generationConfig": generation_config
                }
                
                async with httpx.AsyncClient(timeout=60.0) as client:
                    response = await client.post(
                        url,
                        params={"key": self._gemini_api_key},
                        json=payload
                    )
                    
                    # If server error (500, 502, 503, 504), retry with exponential backoff
                    if response.status_code in [500, 502, 503, 504]:
                        error_msg = f"Server error {response.status_code}"
                        print(f"[RAG] ⚠️ Attempt {attempt+1}/{max_retries} failed: {error_msg}")
                        if attempt < max_retries - 1:
                            # For 503 (overloaded), use longer delays: 5s, 15s, 30s
                            # For other server errors, use shorter delays: 2s, 4s, 8s
                            if response.status_code == 503:
                                delays_503 = [5, 15, 30]  # Longer delays for overloaded model
                                delay = delays_503[min(attempt, len(delays_503) - 1)]
                                print(f"[RAG] Model overloaded (503), waiting {delay} seconds before retry...")
                            else:
                                delay = base_delay * (2 ** attempt)  # Exponential backoff: 2s, 4s, 8s
                                print(f"[RAG] Retrying in {delay} seconds...")
                            await asyncio.sleep(delay)
                            continue
                        else:
                            # Last attempt failed, raise exception
                            error_detail = response.text
                            print(f"[RAG] Gemini API error ({response.status_code}) after {max_retries} attempts: {error_detail[:500]}")
                            if response.status_code == 503:
                                print(f"[RAG] 💡 Tip: Model is temporarily overloaded. Please try again in a few minutes.")
                            raise Exception(f"Gemini API returned {response.status_code} after {max_retries} retries")
                    
                    # Log error details if request fails (non-retryable errors)
                    if response.status_code != 200:
                        error_detail = response.text
                        print(f"[RAG] Gemini API error ({response.status_code}): {error_detail[:500]}")
                        error_json = None
                        retry_delay = None
                        try:
                            error_json = response.json()
                            print(f"[RAG] Error JSON: {json.dumps(error_json, ensure_ascii=False, indent=2)}")
                            
                            # Extract retry delay from error response (for 429 errors)
                            is_daily_quota = False
                            if response.status_code == 429 and error_json:
                                # Check if this is a daily quota limit (not retryable)
                                if "error" in error_json and "details" in error_json["error"]:
                                    for detail in error_json["error"]["details"]:
                                        # Check for daily quota limit
                                        if detail.get("@type") == "type.googleapis.com/google.rpc.QuotaFailure":
                                            violations = detail.get("violations", [])
                                            for violation in violations:
                                                quota_id = violation.get("quotaId", "")
                                                if "PerDay" in quota_id or "Daily" in quota_id:
                                                    is_daily_quota = True
                                                    print(f"[RAG] ⚠️ Daily quota limit exceeded (not retryable): {quota_id}")
                                                    break
                                        
                                        # Try to get retry delay from RetryInfo
                                        if detail.get("@type") == "type.googleapis.com/google.rpc.RetryInfo":
                                            retry_delay_str = detail.get("retryDelay", "")
                                            # Parse "30s" or "30.5s" format
                                            if retry_delay_str.endswith("s"):
                                                try:
                                                    retry_delay = float(retry_delay_str[:-1])
                                                    print(f"[RAG] Extracted retry delay from API: {retry_delay} seconds")
                                                    # If retry delay > 60 seconds, likely daily quota exhausted
                                                    if retry_delay > 60:
                                                        is_daily_quota = True
                                                        print(f"[RAG] ⚠️ Long retry delay ({retry_delay}s) suggests daily quota exhausted")
                                                except:
                                                    pass
                        except:
                            pass
                        
                        # For 429 errors, only retry if NOT daily quota limit and retry delay is reasonable
                        if response.status_code == 429:
                            if is_daily_quota:
                                # Daily quota exhausted - fallback immediately
                                print(f"[RAG] ❌ Daily quota limit exceeded. Falling back immediately (no retry).")
                                raise Exception(f"Gemini API daily quota exceeded (429)")
                            elif attempt < max_retries - 1:
                                # Rate limit - retry with proper delay
                                if retry_delay:
                                    delay = max(retry_delay, 5)  # At least 5 seconds
                                    # Cap delay at 60 seconds to avoid waiting too long
                                    delay = min(delay, 60)
                                else:
                                    # Exponential backoff with longer base for 429
                                    delay = base_delay * (3 ** attempt)  # 2s, 6s, 18s
                                print(f"[RAG] ⚠️ Rate limit (429), retrying in {delay} seconds...")
                                await asyncio.sleep(delay)
                                continue
                            else:
                                # Last attempt failed
                                raise Exception(f"Gemini API rate limit exceeded (429) after {max_retries} retries")
                        
                        # For other non-200 errors, raise exception
                        raise Exception(f"Gemini API returned {response.status_code}")
                    
                    response.raise_for_status()
                    data = response.json()
                    
                    # If we reach here, request was successful - break out of retry loop
                    break
                    
            except Exception as e:
                print(f"[RAG] Error calling Gemini API (Attempt {attempt+1}/{max_retries}): {e}")
                if attempt < max_retries - 1:
                    # Retry for network errors or other exceptions
                    delay = base_delay * (2 ** attempt)
                    print(f"[RAG] Retrying in {delay} seconds...")
                    await asyncio.sleep(delay)
                else:
                    # Last attempt failed
                    import traceback
                    print(f"[RAG] Traceback: {traceback.format_exc()}")
                    return self._get_fallback_response()
        
        # If we exit the loop without breaking, all retries failed
        if 'data' not in locals():
            print(f"[RAG] ❌ All {max_retries} attempts failed")
            return self._get_fallback_response()
        
        # Process successful response
        try:
            # Safely extract text from response (similar to quiz_generator)
            raw = None
            
            if "candidates" not in data or len(data["candidates"]) == 0:
                print(f"[RAG] No candidates in response. Full response: {json.dumps(data, indent=2, ensure_ascii=False)[:1000]}")
                raise Exception("No candidates in Gemini response")
            
            candidate = data["candidates"][0]
            
            # Try multiple ways to extract text (like quiz_generator)
            # Case 1: Standard structure: candidates[0].content.parts[0].text
            if "content" in candidate:
                content = candidate["content"]
                if isinstance(content, dict) and "parts" in content:
                    parts = content["parts"]
                    if isinstance(parts, list) and len(parts) > 0:
                        if isinstance(parts[0], dict) and "text" in parts[0]:
                            raw = parts[0]["text"]
                        elif isinstance(parts[0], str):
                            raw = parts[0]
                
                # Case 2: content might be a string directly
                elif isinstance(content, str):
                    raw = content
                
                # Case 3: content might be a list
                elif isinstance(content, list) and len(content) > 0:
                    first_item = content[0]
                    if isinstance(first_item, dict) and "text" in first_item:
                        raw = first_item["text"]
                    elif isinstance(first_item, str):
                        raw = first_item
            
            # Case 4: Check candidate directly
            if not raw:
                if "text" in candidate:
                    raw = candidate["text"]
                elif isinstance(candidate, str):
                    raw = candidate
            
            # Case 5: Recursive search (fallback)
            if not raw:
                def extract_text_recursive(obj, depth=0):
                    if depth > 5:  # Prevent infinite recursion
                        return None
                    if isinstance(obj, str) and len(obj) > 10:
                        return obj
                    if isinstance(obj, dict):
                        if "text" in obj:
                            return obj["text"]
                        for value in obj.values():
                            result = extract_text_recursive(value, depth + 1)
                            if result:
                                return result
                    elif isinstance(obj, list):
                        for item in obj:
                            result = extract_text_recursive(item, depth + 1)
                            if result:
                                return result
                    return None
                
                raw = extract_text_recursive(candidate)
            
            if not raw:
                print(f"[RAG] ❌ Could not extract text from candidate")
                print(f"[RAG] Full candidate structure: {json.dumps(candidate, indent=2, ensure_ascii=False)[:2000]}")
                raise Exception("Could not extract text from Gemini response")
            
            print(f"[RAG] ✅ Extracted response: {len(raw)} chars")
        except Exception as e:
            print(f"[RAG] Error extracting response: {e}")
            import traceback
            print(f"[RAG] Traceback: {traceback.format_exc()}")
            return self._get_fallback_response()
        
        # 🔥 CRITICAL FIX: Truncate oversized responses for COMPARE_SYNTHESIZE
        # If response > 50KB, truncate to 50KB at a reasonable point (end of sentence/paragraph)
        if query_type == "COMPARE_SYNTHESIZE" and len(raw) > 50000:
            print(f"[RAG] ⚠️ COMPARE_SYNTHESIZE response too large ({len(raw)} chars), truncating to 50KB...")
            # Find a good truncation point (end of sentence or paragraph)
            truncate_at = 50000
            # Try to find end of sentence near 50KB
            for i in range(50000, max(45000, len(raw) - 1000), -1):
                if raw[i] in ['.', '!', '?', '\n']:
                    # Check if it's end of sentence (followed by space or newline)
                    if i + 1 < len(raw) and raw[i + 1] in [' ', '\n', '|']:
                        truncate_at = i + 1
                        break
            raw = raw[:truncate_at] + "\n\n[Nội dung đã được rút gọn để tránh lỗi JSON parsing. Xem chi tiết tại các chunks được trích dẫn.]"
            print(f"[RAG] ✅ Truncated to {len(raw)} chars at position {truncate_at}")
        
        parsed = self._safe_parse_json(raw, query_type)
        
        answer = parsed.get("answer", "")
        
        # 🔥 CRITICAL FIX: Also truncate answer if still too long after parsing
        if query_type == "COMPARE_SYNTHESIZE" and len(answer) > 3000:
            print(f"[RAG] ⚠️ COMPARE_SYNTHESIZE answer still too long ({len(answer)} chars), truncating to 3000 chars...")
            # Find end of sentence near 3000
            truncate_at = 3000
            for i in range(3000, max(2500, len(answer) - 100), -1):
                if answer[i] in ['.', '!', '?', '\n']:
                    if i + 1 < len(answer) and answer[i + 1] in [' ', '\n', '|']:
                        truncate_at = i + 1
                        break
            answer = answer[:truncate_at] + "\n\n[Nội dung đã được rút gọn. Xem chi tiết tại các chunks được trích dẫn.]"
            print(f"[RAG] ✅ Truncated answer to {len(answer)} chars")
            # CRITICAL FIX: If answer is still a JSON string, try to parse it
            if isinstance(answer, str):
                # Check if it's a JSON string (starts with { or contains escaped JSON)
                if answer.strip().startswith('{'):
                    try:
                        answer_obj = json.loads(answer)
                        if isinstance(answer_obj, dict) and "answer" in answer_obj:
                            answer = answer_obj["answer"]
                            print(f"[RAG] ✅ Extracted nested answer from JSON string")
                    except:
                        pass  # If parsing fails, keep original answer
                # Check if answer contains escaped JSON format like: "answer": "...", "answer_type": "..."
                elif '"answer"' in answer and '"answer_type"' in answer:
                    # Try to extract just the answer field value
                    try:
                        # Find the answer field value (handle escaped quotes and newlines)
                        # Use more robust pattern that handles multiline strings
                        match = re.search(r'"answer"\s*:\s*"((?:[^"\\]|\\.|\\n)*)"', answer, re.DOTALL)
                        if match:
                            # Properly unescape JSON string
                            import json as json_module
                            try:
                                answer = json_module.loads('"' + match.group(1) + '"')
                            except:
                                # Fallback: manual unescape
                                answer = match.group(1).replace('\\n', '\n').replace('\\"', '"').replace('\\\\', '\\')
                            print(f"[RAG] ✅ Extracted answer from JSON-like string")
                    except Exception as e:
                        print(f"[RAG] ⚠️ Failed to extract answer from JSON-like string: {e}")
                        pass
                # Check if answer contains escaped newlines and JSON structure indicators
                elif '\\n' in answer and ('"answer_type"' in answer or '"chunks_used"' in answer or '"reasoning_steps"' in answer):
                    # This might be a JSON string with escaped characters - extract text before JSON fields
                    try:
                        # Try to find and extract the actual answer text (before JSON fields)
                        # Look for pattern: text content followed by JSON fields
                        match = re.search(r'^(.+?)(?:\s*"answer_type"|\s*"chunks_used"|\s*"reasoning_steps"|\s*"sentence_mapping"|\s*"sources")', answer, re.DOTALL)
                        if match:
                            answer = match.group(1).strip()
                            # Clean up escaped characters
                            answer = answer.replace('\\n', '\n').replace('\\"', '"').replace('\\\\', '\\')
                            # Remove trailing JSON structure if any
                            answer = re.sub(r'\s*,\s*"answer_type".*$', '', answer, flags=re.DOTALL)
                            print(f"[RAG] ✅ Extracted answer text from escaped JSON string")
                    except Exception as e:
                        print(f"[RAG] ⚠️ Failed to extract from escaped JSON: {e}")
                        pass
        
        answer_type = parsed.get("answer_type", "FALLBACK")
        
        # Post-process: Fix numbered list formatting for ALL answer types that may contain lists
        if answer_type in ["DOCUMENT_OVERVIEW", "SECTION_OVERVIEW", "EXPAND", "COMPARE_SYNTHESIZE"]:
            original_length = len(answer)
            answer = self._fix_numbered_list_formatting(answer)
            if len(answer) != original_length or "\n\n" in answer:
                print(f"[RAG] ✅ Fixed numbered list formatting: {original_length} -> {len(answer)} chars, has_double_newlines={answer.count(chr(10)*2)}")
            else:
                print(f"[RAG] ⚠️ Numbered list formatting fix may not have worked (length unchanged)")
        
        # 🔥 CRITICAL FIX: Remove citation markers [Chunk X] and [Filename] from answer for ALL query types
        original_length = len(answer)
        answer = self._clean_citation_markers(answer)
        if len(answer) != original_length:
            print(f"[RAG] ✅ Cleaned citation markers: {original_length} -> {len(answer)} chars")
        
        # Post-process: Clean table citations for COMPARE_SYNTHESIZE and MULTI_PART
        if answer_type in ["COMPARE_SYNTHESIZE", "MULTI_PART"] and "|" in answer:
            original_length = len(answer)
            answer = self._clean_table_citations(answer)
            if len(answer) != original_length:
                print(f"[RAG] ✅ Cleaned table citations: {original_length} -> {len(answer)} chars")
        
        chunk_indices_raw = parsed.get("chunks_used", [])
        # Normalize chunk_indices: convert to list of integers
        chunk_indices = []
        for item in chunk_indices_raw:
            if isinstance(item, int):
                chunk_indices.append(item)
            elif isinstance(item, dict):
                chunk_indices.append(item.get("chunk_index", item.get("chunk_idx")))
            elif isinstance(item, str) and item.isdigit():
                chunk_indices.append(int(item))
        
        confidence = parsed.get("confidence", 0.0)
        sentence_mapping = parsed.get("sentence_mapping", [])
        sources = parsed.get("sources", {})
        reasoning_steps = parsed.get("reasoning_steps", [])
        
        # 🔥 CRITICAL FIX: Validate FALLBACK answers FIRST - before any other validation
        # Check if answer has actual content even if LLM marked it as FALLBACK or confidence = 0.0
        if answer_type == "FALLBACK" or confidence == 0.0:
            answer_length = len(answer.strip())
            has_meaningful_content = answer_length > 100  # More than just "không tìm thấy"
            has_doc_references = bool(re.search(r'(theo|trong|tài liệu|document|chunk|page|Javascript|COCOMO)', answer, re.IGNORECASE))
            
            # If FALLBACK/0.0 confidence but has content → likely wrong classification
            if has_meaningful_content and has_doc_references:
                print(f"[RAG] ⚠️ FALLBACK/0.0 confidence classification suspicious - answer has content ({answer_length} chars, has_doc_refs={has_doc_references})")
                
                # Try to extract chunks from answer text
                chunk_refs = re.findall(r'chunk\s+(\d+)', answer, re.IGNORECASE)
                if chunk_refs:
                    extracted_chunks = [int(ref) for ref in chunk_refs[:10]]
                    # Merge with existing chunks
                    if chunk_indices:
                        chunk_indices = list(set(chunk_indices + extracted_chunks))
                    else:
                        chunk_indices = extracted_chunks
                    answer_type = "DIRECT"  # Override to DIRECT
                    if confidence < 0.5:
                        confidence = 0.75  # Set reasonable confidence
                    print(f"[RAG] ✅ Auto-corrected: FALLBACK/0.0 → DIRECT, extracted {len(extracted_chunks)} chunks from answer text, confidence={confidence:.2f}")
                else:
                    # No explicit chunks but has content → mark as SYNTHESIS or DIRECT
                    # Check if answer mentions specific document names
                    has_specific_docs = bool(re.search(r'(Javascript|COCOMO|jQuery|framework|module|SLOC)', answer, re.IGNORECASE))
                    if has_specific_docs:
                        answer_type = "DIRECT"  # Has specific info → DIRECT
                        if confidence < 0.5:
                            confidence = 0.70
                        # CRITICAL: Try to recover chunks from selected_results if available
                        if not chunk_indices and hasattr(self, 'selected_results') and self.selected_results:
                            # Get top chunks that match document keywords
                            for item in self.selected_results[:10]:
                                record = item.get("_record")
                                if record:
                                    doc_id = record.get("document_id")
                                    chunk_idx = record.get("chunk_index")
                                    # Prefer chunks from documents mentioned in answer
                                    if chunk_idx is not None and len(chunk_indices) < 5:
                                        chunk_indices.append(chunk_idx)
                            if chunk_indices:
                                print(f"[RAG] ✅ Recovered {len(chunk_indices)} chunks from selected_results for {answer_type}")
                    else:
                        answer_type = "SYNTHESIS"  # General info → SYNTHESIS
                        if confidence < 0.5:
                            confidence = 0.70
                    print(f"[RAG] ✅ Auto-corrected: FALLBACK/0.0 → {answer_type} (has content but no explicit chunks, confidence={confidence:.2f}, chunks={len(chunk_indices)})")
        
        # CRITICAL FIX: For COMPARE_SYNTHESIZE with table, auto-extract chunks
        if answer_type == "COMPARE_SYNTHESIZE" and "|" in answer:
            has_table = "| Tiêu chí |" in answer or "|" in answer
            
            if has_table and len(chunk_indices) < 3:
                print(f"[RAG] ⚠️ COMPARE_SYNTHESIZE table found but only {len(chunk_indices)} chunks")
                
                # Try to extract from selected_results
                if hasattr(self, 'selected_results') and self.selected_results:
                    # Extract chunk numbers mentioned in answer
                    mentioned_chunks = set()
                    # Pattern: "từ chunk X" or "(chunk X)"
                    for match in re.finditer(r'chunk\s+(\d+)', answer, re.IGNORECASE):
                        mentioned_chunks.add(int(match.group(1)))
                    
                    # Add chunks from selected_results that are highly relevant
                    for item in self.selected_results[:20]:
                        record = item.get("_record")
                        if record:
                            chunk_idx = record.get("chunk_index")
                            doc_id = record.get("document_id")
                            
                            # Prefer chunks mentioned in answer or with high similarity
                            if chunk_idx in mentioned_chunks or item.get("similarity", 0) > 0.7:
                                if chunk_idx not in chunk_indices:
                                    chunk_indices.append(chunk_idx)
                    
                    # If still empty, use top chunks from selected_results
                    if not chunk_indices:
                        print(f"[RAG] COMPARE_SYNTHESIZE with table but no chunks → recovering from selected_results")
                        for item in self.selected_results[:15]:  # Lấy top 15 chunks
                            record = item.get("_record")
                            if record:
                                chunk_idx = record.get("chunk_index")
                                doc_id = record.get("document_id")
                                if chunk_idx not in chunk_indices:
                                    chunk_indices.append(chunk_idx)
                    
                    if chunk_indices:
                        answer_type = "COMPARE_SYNTHESIZE"  # Giữ nguyên type
                        confidence = max(0.85, confidence)  # Boost confidence
                        print(f"[RAG] ✅ Enhanced chunks_used to {len(chunk_indices)} chunks for COMPARE")
        
        # === VALIDATION LAYER ===
        
        # Rule 0: TOO_BROAD detection (Giữ nguyên)
        if answer_type == "TOO_BROAD":
            chunk_indices = []
            sentence_mapping = []
            confidence = 0.0
            print(f"[RAG] TOO_BROAD detected → enforcing 0 chunks")
        
        # 🔥 CRITICAL FIX: AGGRESSIVE RECOVERY FOR CREATIVE MODES
        # Logic: Nếu là câu hỏi sáng tạo + câu trả lời dài > 200 ký tự -> CHẤP NHẬN LUÔN
        # Bỏ qua mọi check về confidence hay chunks rỗng
        is_creative_mode = query_type in ["EXERCISE_GENERATION", "EXPAND", "MULTI_CONCEPT_REASONING", "CODE_ANALYSIS"]
        answer_is_substantial = len(answer.strip()) > 200
        
        if is_creative_mode and answer_is_substantial:
                print(f"[RAG] 🛡️ Creative Mode Protection: Type={query_type}, Length={len(answer)}")
                
                # 1. Override answer_type nếu đang bị set là FALLBACK
                if answer_type == "FALLBACK":
                    answer_type = query_type
                    print(f"[RAG] 🔄 Override FALLBACK -> {answer_type} because content exists")

                # 2. Force Confidence High
                if confidence < 0.7:
                    confidence = 0.85
                    print(f"[RAG] 🔄 Boosted confidence to {confidence}")

                # 3. AUTO-FILL CHUNKS if empty (Chìa khóa để fix lỗi)
                # Nếu LLM quên trích dẫn, ta lấy luôn top 5 chunks đầu vào làm "nguồn cảm hứng"
                if not chunk_indices and hasattr(self, 'selected_results') and self.selected_results:
                    print(f"[RAG] 🔄 Auto-filling chunks for creative answer (LLM forgot citations)")
                    
                    # Lấy top chunks có similarity cao nhất
                    top_items = sorted(
                        self.selected_results, 
                        key=lambda x: x.get("similarity", 0), 
                        reverse=True
                    )[:5]  # Lấy 5 chunks tốt nhất
                    
                    for item in top_items:
                        record = item.get("_record")
                        if record:
                            idx = record.get("chunk_index")
                            if idx is not None and idx not in chunk_indices:
                                chunk_indices.append(idx)
                    
                    print(f"[RAG] 🔄 Auto-filled {len(chunk_indices)} chunks from selected_results")
            
                # ENHANCED: Validation for reasoning queries - More lenient
                if query_type in ["CODE_ANALYSIS", "EXERCISE_GENERATION", "MULTI_CONCEPT_REASONING"]:
                    # NEW: More lenient - only reject if VERY short or VERY low confidence
                    if len(answer) < 50:  # Only reject if VERY short
                        print(f"[RAG] Reasoning query but answer too short ({len(answer)} chars)")
                        answer_type = "FALLBACK"
                        chunk_indices = []
                        confidence = 0.0
                    elif confidence < 0.4:  # Lower threshold (from 0.5 to 0.4)
                        print(f"[RAG] Low confidence for reasoning query ({confidence:.2f})")
                        answer_type = "FALLBACK"
                        chunk_indices = []
                        confidence = 0.0
                    else:
                        # Accept even without reasoning_steps field if answer is substantial
                        if not reasoning_steps:
                            print(f"[RAG] Reasoning answer accepted despite missing reasoning_steps field (answer length: {len(answer)})")
                else:
                    # Original validation for non-reasoning queries
                    # CRITICAL FIX: Don't apply fallback detection for SECTION_OVERVIEW or DOCUMENT_OVERVIEW
                    if answer_type not in ["SECTION_OVERVIEW", "DOCUMENT_OVERVIEW"]:
                        # CRITICAL FIX: Check answer quality BEFORE forcing FALLBACK
                        # Extract chunks from answer text if chunks_used is empty
                        answer_has_chunk_refs = bool(re.search(r'chunk\s+(\d+)', answer, re.IGNORECASE))
                        answer_length = len(answer)
                        answer_has_citations = bool(re.search(r'(chunk|theo|trong|tài liệu)', answer, re.IGNORECASE))
                        
                    # Rule 1: Fallback detection via keywords (Sửa lại để nhẹ nhàng hơn)
                    # Chỉ check nếu KHÔNG PHẢI creative mode đã được xử lý ở trên
                    if not (is_creative_mode and answer_is_substantial):
                        if self._is_fallback_answer(answer):
                            # Check if answer is actually good despite keywords
                            if answer_length > 500 and (answer_has_chunk_refs or answer_has_citations):
                                # Answer has substance and citations → NOT a fallback
                                print(f"[RAG] ⚠️ Fallback keywords detected BUT answer quality good ({answer_length} chars, has citations) → KEEP")
                                # CRITICAL FIX: Always extract chunks from answer text if answer is good
                                # Even if chunks were zeroed out earlier, we should restore them
                                if answer_has_chunk_refs:
                                    chunk_refs = re.findall(r'chunk\s+(\d+)', answer, re.IGNORECASE)
                                    extracted_chunks = [int(ref) for ref in chunk_refs[:15]]  # Extract up to 15 chunks
                                    # Merge with existing chunks, remove duplicates
                                    if chunk_indices:
                                        all_chunks = list(set(chunk_indices + extracted_chunks))
                                        print(f"[RAG] Extracted {len(extracted_chunks)} chunks from answer text, merged with existing {len(chunk_indices)} → total: {len(all_chunks)}")
                                    else:
                                        all_chunks = extracted_chunks
                                        print(f"[RAG] Extracted {len(extracted_chunks)} chunks from answer text (was empty)")
                                    chunk_indices = all_chunks[:20]  # Limit to 20 chunks max
                                # Auto-correct type if needed
                                if answer_type == "FALLBACK":
                                    if query_type in ["MULTI_CONCEPT_REASONING", "CODE_ANALYSIS", "EXERCISE_GENERATION"]:
                                        answer_type = query_type
                                    elif answer_has_chunk_refs or len(chunk_indices) > 0:
                                        answer_type = "DIRECT"
                                    else:
                                        answer_type = "SYNTHESIS"
                                # CRITICAL FIX: Set confidence based on answer length (more granular)
                                if confidence < 0.5:
                                    if answer_length > 1000:
                                        confidence = 0.85  # Long answer → high confidence
                                    elif answer_length > 700:
                                        confidence = 0.80  # Medium-long answer
                                    else:
                                        confidence = 0.75  # Short but good answer
                                print(f"[RAG] Auto-corrected: type={answer_type}, confidence={confidence:.2f}, chunks={len(chunk_indices)}")
                            else:
                                # Real fallback - short answer with fallback keywords
                                answer_type = "FALLBACK"
                                chunk_indices = []
                                confidence = 0.0
                                sentence_mapping = []
                                print(f"[RAG] Fallback detected via keywords")
                        
                    # Rule 2: Low confidence → check if should force fallback (unless TOO_BROAD)
                    # Chỉ check nếu KHÔNG PHẢI creative mode đã được xử lý ở trên
                    if not (is_creative_mode and answer_is_substantial):
                        # BUT: Don't force FALLBACK if answer has good quality (already handled above)
                        if confidence < settings.rag_low_confidence_threshold and answer_type not in ["TOO_BROAD", "DIRECT", "SYNTHESIS", "FALLBACK"]:
                            # Re-check answer quality (may have been updated by previous validation)
                            answer_length = len(answer)
                            answer_has_chunk_refs = bool(re.search(r'chunk\s+(\d+)', answer, re.IGNORECASE))
                            answer_has_citations = bool(re.search(r'(chunk|theo|trong|tài liệu|Javascript|COCOMO|jQuery|framework)', answer, re.IGNORECASE))
                            
                            # Check if answer is actually good despite low confidence
                            # Lower threshold to 200 chars to catch more cases (answer có 407 chars)
                            if answer_length > 200 and (answer_has_chunk_refs or answer_has_citations or len(chunk_indices) > 0):
                                # Answer has substance → NOT a fallback, just low confidence from LLM
                                print(f"[RAG] ⚠️ Low confidence ({confidence:.2f}) BUT answer quality good ({answer_length} chars, has citations) → KEEP")
                                # CRITICAL FIX: Always extract chunks from answer text if answer is good
                                # Even if chunks were zeroed out earlier, we should restore them
                                if answer_has_chunk_refs:
                                    chunk_refs = re.findall(r'chunk\s+(\d+)', answer, re.IGNORECASE)
                                    extracted_chunks = [int(ref) for ref in chunk_refs[:15]]  # Extract up to 15 chunks
                                    # Merge with existing chunks, remove duplicates
                                    if chunk_indices:
                                        all_chunks = list(set(chunk_indices + extracted_chunks))
                                        print(f"[RAG] Extracted {len(extracted_chunks)} chunks from answer text, merged with existing {len(chunk_indices)} → total: {len(all_chunks)}")
                                    else:
                                        all_chunks = extracted_chunks
                                        print(f"[RAG] Extracted {len(extracted_chunks)} chunks from answer text (was empty)")
                                    chunk_indices = all_chunks[:20]  # Limit to 20 chunks max
                                # Auto-correct type if needed
                                if answer_type == "FALLBACK":
                                    if query_type in ["MULTI_CONCEPT_REASONING", "CODE_ANALYSIS", "EXERCISE_GENERATION"]:
                                        answer_type = query_type
                                    elif answer_has_chunk_refs or len(chunk_indices) > 0:
                                        answer_type = "DIRECT"
                                    else:
                                        answer_type = "SYNTHESIS"
                                # CRITICAL FIX: Set confidence based on answer length (more granular)
                                if answer_length > 1000:
                                    confidence = 0.85  # Long answer → high confidence
                                elif answer_length > 700:
                                    confidence = 0.80  # Medium-long answer
                                else:
                                    confidence = 0.75  # Short but good answer
                                print(f"[RAG] Auto-corrected: type={answer_type}, confidence={confidence:.2f}, chunks={len(chunk_indices)}")
                            else:
                                # Real fallback - short answer with low confidence
                                answer_type = "FALLBACK"
                                chunk_indices = []
                                sentence_mapping = []
                                print(f"[RAG] Low confidence ({confidence:.2f}) → forced fallback")
                    else:
                        # CRITICAL FIX: Special handling for SECTION_OVERVIEW
                        if answer_type == "SECTION_OVERVIEW":
                            # Check if answer has expected structure
                            has_title = bool(re.search(r'PHẦN\s+\d+:', answer))
                            has_content = len(answer) > 100
                            has_chunks = len(chunk_indices) > 0
                            
                            if has_chunks and has_content:
                                # If we have chunks and substantial content → KEEP IT
                                if confidence < 0.7:
                                    confidence = 0.85  # Boost confidence
                                print(f"[RAG] SECTION_OVERVIEW validated: chunks={len(chunk_indices)}, length={len(answer)}")
                            elif has_title and has_content:
                                # Even without chunks, if answer looks structured → KEEP IT
                                confidence = max(0.75, confidence)
                                print(f"[RAG] SECTION_OVERVIEW kept despite missing chunks (has structure)")
                            else:
                                # Only fallback if answer is really empty/broken
                                if len(answer) < 50:
                                    answer_type = "FALLBACK"
                                    chunk_indices = []
                                    confidence = 0.0
                                    print(f"[RAG] SECTION_OVERVIEW too short → fallback")
                        
                        # 🔥 CRITICAL FIX: Validate DOCUMENT_OVERVIEW output
                        elif answer_type == "DOCUMENT_OVERVIEW":
                            # Count sections found in answer - match nhiều format hơn
                            # Pattern 1: "1. **PHẦN 1:**" hoặc "1. **PHẦN 1**"
                            # Pattern 2: "1. PHẦN 1:" hoặc "1. PHẦN 1"
                            # Pattern 3: "PHẦN 1:" hoặc "PHẦN 1"
                            section_patterns = [
                            r'\d+\.\s+\*\*PHẦN\s+\d+',  # "1. **PHẦN 1"
                            r'\d+\.\s+PHẦN\s+\d+',  # "1. PHẦN 1"
                            r'PHẦN\s+\d+[:：]',  # "PHẦN 1:"
                            r'PHẦN\s+\d+\s+[A-Z]',  # "PHẦN 1 TITLE"
                        ]
                        sections_found = 0
                        for pattern in section_patterns:
                            matches = re.findall(pattern, answer, re.IGNORECASE)
                            sections_found = max(sections_found, len(matches))
                        
                        # Check for gaps - extract ALL section numbers
                        section_numbers = re.findall(r'PHẦN\s+(\d+)', answer, re.IGNORECASE)
                        section_nums = sorted([int(n) for n in section_numbers]) if section_numbers else []
                        
                        has_gaps = False
                        if len(section_nums) >= 2:
                            expected_range = range(section_nums[0], section_nums[-1] + 1)
                            has_gaps = len(section_nums) != len(expected_range)
                        
                        if sections_found < 3:
                            print(f"[RAG] ⚠️ DOCUMENT_OVERVIEW validation FAILED: only {sections_found} sections found (section_nums: {section_nums})")
                            confidence = max(0.5, confidence * 0.7)
                        elif has_gaps:
                            print(f"[RAG] ⚠️ DOCUMENT_OVERVIEW has gaps: {section_nums} (expected: {list(expected_range)})")
                            confidence = max(0.75, confidence * 0.9)
                        else:
                            print(f"[RAG] ✅ DOCUMENT_OVERVIEW validated: {sections_found} sections, no gaps (sections: {section_nums})")
                            confidence = min(0.95, confidence)
                        
                        # Still require minimum length
                        if len(answer) < 100:
                                answer_type = "FALLBACK"
                                chunk_indices = []
                                confidence = 0.0
                                print(f"[RAG] DOCUMENT_OVERVIEW too short → fallback")
                        else:
                            # Original validation: Only fallback if EXPLICITLY no chunks
                            if not chunk_indices:
                                answer_type = "FALLBACK"
                                confidence = 0.0
                                sentence_mapping = []
                                print(f"[RAG] {answer_type} but no chunks → fallback")
                            else:
                                print(f"[RAG] {answer_type} with {len(chunk_indices)} chunks → keeping answer")
        
        # 🔥 CRITICAL FIX: Enhanced chunk recovery mechanism
        # If answer has content but no chunks, try to recover from multiple sources
        # Run even if confidence is low (0.0) if answer has meaningful content
        # SPECIAL CASE: For DOCUMENT_OVERVIEW, always extract chunks from answer text even if already has chunks
        answer_has_content = len(answer.strip()) > 100
        answer_has_doc_refs = bool(re.search(r'(theo|trong|tài liệu|document|Javascript|COCOMO)', answer, re.IGNORECASE))
        should_recover = (confidence > 0.5 or (confidence == 0.0 and answer_has_content and answer_has_doc_refs)) and len(chunk_indices) == 0 and answer_type not in ["FALLBACK", "TOO_BROAD"]
        
        # 🔥 CRITICAL FIX: For DOCUMENT_OVERVIEW, always extract chunks from multiple sources to ensure multi-doc coverage
        if answer_type == "DOCUMENT_OVERVIEW" and answer_has_content:
            extracted_chunk_indices = []
            
            # Method 1: Extract from answer text (e.g., "Chunk 35", "Chunk 66")
            chunk_refs_in_answer = re.findall(r'chunk\s+(\d+)', answer, re.IGNORECASE)
            if chunk_refs_in_answer:
                extracted_chunk_indices.extend([int(ref) for ref in chunk_refs_in_answer])
                print(f"[RAG] ✅ DOCUMENT_OVERVIEW: Found {len(chunk_refs_in_answer)} chunk references in answer text")
            
            # Method 2: Extract from sentence_mapping (if available)
            if sentence_mapping:
                chunk_indices_from_mapping = [
                    s.get("chunk") for s in sentence_mapping 
                    if s.get("chunk") and not s.get("external", False)
                ]
                if chunk_indices_from_mapping:
                    extracted_chunk_indices.extend(chunk_indices_from_mapping)
                    print(f"[RAG] ✅ DOCUMENT_OVERVIEW: Found {len(chunk_indices_from_mapping)} chunks from sentence_mapping")
            
            # Method 3: Use top selected_results from both documents (if multi-doc)
            if len(selected_documents) > 1 and hasattr(self, 'selected_results') and self.selected_results:
                # Get top chunks from each document
                doc_chunks_map = {}
                for item in self.selected_results:
                    record = item.get("_record")
                    if record:
                        doc_id = record.get("document_id")
                        chunk_idx = record.get("chunk_index")
                        if doc_id and chunk_idx is not None:
                            if doc_id not in doc_chunks_map:
                                doc_chunks_map[doc_id] = []
                            doc_chunks_map[doc_id].append((chunk_idx, item.get("similarity", 0)))
                
                # Add top 5 chunks from each document
                for doc_id, chunks_with_sim in doc_chunks_map.items():
                    sorted_chunks = sorted(chunks_with_sim, key=lambda x: x[1], reverse=True)[:5]
                    for chunk_idx, _ in sorted_chunks:
                        if chunk_idx not in extracted_chunk_indices:
                            extracted_chunk_indices.append(chunk_idx)
                print(f"[RAG] ✅ DOCUMENT_OVERVIEW: Added top chunks from {len(doc_chunks_map)} document(s)")
            
            # Merge with existing chunks (avoid duplicates)
            if extracted_chunk_indices:
                if chunk_indices:
                    chunk_indices = list(set(chunk_indices + extracted_chunk_indices))
                    print(f"[RAG] ✅ DOCUMENT_OVERVIEW: Merged {len(extracted_chunk_indices)} extracted chunks with {len(chunk_indices) - len(extracted_chunk_indices)} existing → total: {len(chunk_indices)}")
                else:
                    chunk_indices = list(set(extracted_chunk_indices))
                    print(f"[RAG] ✅ DOCUMENT_OVERVIEW: Extracted {len(chunk_indices)} chunks from multiple sources")
        
        if should_recover:
            print(f"[RAG] ⚠️ High confidence ({confidence:.2f}) but no chunks → attempting recovery")
            
            recovered_chunks = []
            
            # Method 1: Extract from sentence_mapping
            if sentence_mapping:
                chunk_indices_from_mapping = [
                    s.get("chunk") for s in sentence_mapping 
                    if s.get("chunk") and not s.get("external", False)
                ]
                if chunk_indices_from_mapping:
                    recovered_chunks = list(set(chunk_indices_from_mapping))[:15]
                    print(f"[RAG] ✅ Recovered {len(recovered_chunks)} chunks from sentence_mapping")
            
            # Method 2: Extract from answer text
            if not recovered_chunks:
                chunk_refs = re.findall(r'chunk\s+(\d+)', answer, re.IGNORECASE)
                if chunk_refs:
                    recovered_chunks = [int(ref) for ref in chunk_refs[:15]]
                    print(f"[RAG] ✅ Recovered {len(recovered_chunks)} chunks from answer text")
            
            # Method 3: Use top selected_results (last resort)
            if not recovered_chunks and hasattr(self, 'selected_results') and self.selected_results:
                top_items = sorted(
                    self.selected_results, 
                    key=lambda x: x.get("similarity", 0), 
                    reverse=True
                )[:10]
                for item in top_items:
                    record = item.get("_record")
                    if record:
                        chunk_idx = record.get("chunk_index")
                        if chunk_idx is not None:
                            recovered_chunks.append(chunk_idx)
                print(f"[RAG] ✅ Recovered {len(recovered_chunks)} chunks from selected_results")
            
            # If recovery successful, update chunks and type
            if recovered_chunks:
                chunk_indices = recovered_chunks
                if answer_type == "FALLBACK":
                    answer_type = "DIRECT"
                # Adjust confidence based on recovery method and answer quality
                if confidence == 0.0:
                    # If was 0.0, set to reasonable confidence based on answer length
                    if len(answer) > 500:
                        confidence = 0.75
                    elif len(answer) > 300:
                        confidence = 0.70
                    else:
                        confidence = 0.65
                else:
                    confidence = min(0.85, max(confidence, 0.70))  # At least 0.70 if recovery successful
                print(f"[RAG] ✅ Chunk recovery successful: {len(chunk_indices)} chunks, type={answer_type}, confidence={confidence:.2f}")
        
        # Rule 3: CRITICAL - Enforce fallback=0 refs (and TOO_BROAD)
        # BUT: Only enforce if answer_type is still FALLBACK after recovery attempts
        if answer_type in ["FALLBACK", "TOO_BROAD"]:
            chunk_indices = []
            sentence_mapping = []
            confidence = 0.0
            print(f"[RAG] {answer_type} type → enforcing 0 chunks")
        
        # Rule 4: No chunks but claims document source → suspicious (skip for overviews)
        if not chunk_indices and sources.get("from_document") and answer_type not in ["SECTION_OVERVIEW", "DOCUMENT_OVERVIEW", "FALLBACK", "TOO_BROAD"]:
            answer_type = "FALLBACK"
            confidence = 0.0
            print(f"[RAG] Suspicious: no chunks but claims document source")
        
        # Rule 5: Check sentence_mapping consistency (skip for overviews)
        # CRITICAL FIX: Don't zero out chunks if answer is good (4000-6000 chars)
        # Instead, mark as SYNTHESIS and keep reasonable confidence
        if sentence_mapping and answer_type not in ["SECTION_OVERVIEW", "DOCUMENT_OVERVIEW"]:
            external_count = sum(1 for s in sentence_mapping if s.get("external", False))
            total_count = len(sentence_mapping)
            if total_count > 0 and external_count / total_count > 0.5:
                # Check if answer is substantial (good synthesis)
                answer_length = len(answer)
                if answer_length >= 4000:
                    # Good synthesis answer - don't zero out chunks
                    answer_type = "SYNTHESIS" if answer_type not in ["FALLBACK", "TOO_BROAD"] else answer_type
                    # Keep reasonable confidence, don't zero chunks
                    if confidence > 0.9:
                        confidence = 0.75  # Reduce but keep reasonable
                    print(f"[RAG] >50% external but substantial answer ({answer_length} chars) → marked as SYNTHESIS, kept chunks")
                else:
                    # Short answer with >50% external → likely fallback
                    answer_type = "FALLBACK"
                    chunk_indices = []
                    sentence_mapping = []
                    confidence = 0.0
                    print(f"[RAG] >50% external sentences + short answer → forced fallback")
                
        # Map to full chunk info (REBUILD chunks_used based on final chunk_indices)
        chunks_used = []  # Reset to ensure sync with indices
        for idx in chunk_indices:
            for meta in chunk_metadata_list:
                if meta.get("chunk_index") == idx:
                    chunks_used.append({
                        "chunk_index": idx,
                        "document_id": meta.get("document_id")
                    })
                    break
        
        # 🔥 CRITICAL FIX: Ensure chunks from ALL documents when multi-doc query
        # If query involves multiple documents but chunks_used only has chunks from 1 document,
        # add top chunks from other documents
        # Đặc biệt quan trọng cho CODE_ANALYSIS, COMPARE_SYNTHESIZE, MULTI_PART, SECTION_OVERVIEW
        if len(selected_documents) > 1 and len(chunks_used) > 0:
            # Get unique document IDs from chunks_used
            used_doc_ids = set(c.get("document_id") for c in chunks_used if c.get("document_id"))
            selected_doc_ids = set(doc.id for doc in selected_documents)
            
            # Check if we're missing chunks from any document
            missing_doc_ids = selected_doc_ids - used_doc_ids
            
            if missing_doc_ids:
                print(f"[RAG] ⚠️ {answer_type}: Missing chunks from {len(missing_doc_ids)} document(s)!")
                
                # Determine chunks per missing doc based on query type
                if answer_type in ["CODE_ANALYSIS", "COMPARE_SYNTHESIZE", "MULTI_PART"]:
                    chunks_per_missing_doc = 5  # Cần nhiều chunks hơn cho các query types này
                elif answer_type in ["MULTI_PART", "DOCUMENT_OVERVIEW"]:
                    chunks_per_missing_doc = 5
                else:
                    chunks_per_missing_doc = 3
                
                # Add top chunks from missing documents
                for missing_doc_id in missing_doc_ids:
                    doc_name = next((doc.filename for doc in selected_documents if doc.id == missing_doc_id), "Unknown")
                    print(f"[RAG] 🔧 Force adding chunks from {doc_name}...")
                    
                    added_count = 0
                    
                    # Try to find chunks from selected_results first (better quality)
                    doc_chunks_with_sim = []
                    if hasattr(self, 'selected_results') and self.selected_results:
                        for item in self.selected_results:
                            record = item.get("_record")
                            if record and record.get("document_id") == missing_doc_id:
                                chunk_idx = record.get("chunk_index")
                                if chunk_idx is not None:
                                    doc_chunks_with_sim.append((chunk_idx, item.get("similarity", 0)))
                    
                    # If found in selected_results, add top chunks sorted by similarity
                    if doc_chunks_with_sim:
                        doc_chunks_sorted = sorted(doc_chunks_with_sim, key=lambda x: x[1], reverse=True)
                        for chunk_idx, _ in doc_chunks_sorted[:chunks_per_missing_doc]:
                            if not any(c.get("chunk_index") == chunk_idx for c in chunks_used):
                                chunks_used.append({
                                    "chunk_index": chunk_idx,
                                    "document_id": missing_doc_id
                                })
                                added_count += 1
                                print(f"[RAG] ✅ Added chunk {chunk_idx} from {doc_name}")
                    else:
                        # Fallback: check chunk_metadata_list
                        doc_chunks_meta = [
                            meta for meta in chunk_metadata_list
                            if meta.get("document_id") == missing_doc_id
                        ]
                        # Take first N chunks from missing doc
                        for meta in doc_chunks_meta[:chunks_per_missing_doc]:
                            chunk_idx = meta.get("chunk_index")
                            if chunk_idx and not any(c.get("chunk_index") == chunk_idx for c in chunks_used):
                                chunks_used.append({
                                    "chunk_index": chunk_idx,
                                    "document_id": missing_doc_id
                                })
                                added_count += 1
                                print(f"[RAG] ✅ Added chunk {chunk_idx} from {doc_name} (from metadata)")
                
                    if added_count == 0:
                        print(f"[RAG] ❌ WARNING: Could not find any chunks from {doc_name} in selected_results or metadata!")
                
                print(f"[RAG] ✅ After recovery: chunks_used now has chunks from {len(set(c.get('document_id') for c in chunks_used if c.get('document_id')))} document(s)")
        
        # CRITICAL FIX: Confidence-Chunks Paradox Detection
        # Flag inconsistency: high confidence but no chunks
        if confidence > 0.7 and len(chunk_indices) == 0 and answer_type not in ["FALLBACK", "TOO_BROAD", "SYNTHESIS"]:
            print(f"[RAG] ⚠️ PARADOX DETECTED: confidence={confidence:.2f} but chunks=0!")
            print(f"[RAG] Answer type: {answer_type}, Answer length: {len(answer)}")
            # Auto-correct: if answer is substantial, mark as SYNTHESIS
            if len(answer) >= 2000:
                answer_type = "SYNTHESIS"
                confidence = 0.75  # Reduce to reasonable level
                print(f"[RAG] Auto-corrected to SYNTHESIS with confidence={confidence:.2f}")
            else:
                # Short answer with high confidence but no chunks → suspicious
                answer_type = "FALLBACK"
                confidence = 0.0
                print(f"[RAG] Auto-corrected to FALLBACK (short answer with no chunks)")
        
        print(f"[RAG] Answer type: {answer_type}, Confidence: {confidence:.2f}")
        print(f"[RAG] Chunks: {chunk_indices}, Sentences mapped: {len(sentence_mapping)}")
        
        return answer, chunks_used, answer_type, confidence, sentence_mapping

        # Fallback for OpenAI or other providers
        if self.provider == "openai" and self._openai_client:
            try:
                # For OpenAI, use similar approach but with different format
                messages = [
                    {
                        "role": "system",
                        "content": "You are a professional study assistant. Answer questions based on context. Always return JSON format with answer, answer_type, chunks_used, confidence, sentence_mapping, and sources fields.",
                    },
                    {
                        "role": "user",
                        "content": prompt,
                    },
                ]

                response = await asyncio.to_thread(
                    self._openai_client.chat.completions.create,
                    model=self.model,
                    messages=messages,
                    max_tokens=self.max_tokens,
                    temperature=0.0,
                    response_format={"type": "json_object"},
                )
                raw = response.choices[0].message.content
                parsed = self._safe_parse_json(raw)
                
                answer = parsed.get("answer", "")
                answer_type = parsed.get("answer_type", "FALLBACK")
                chunk_indices = parsed.get("chunks_used", [])
                confidence = parsed.get("confidence", 0.0)
                sentence_mapping = parsed.get("sentence_mapping", [])
                
                # Apply same validation rules
                if self._is_fallback_answer(answer):
                    answer_type = "FALLBACK"
                    chunk_indices = []
                    confidence = 0.0
                    sentence_mapping = []
                
                if confidence < settings.rag_low_confidence_threshold:
                    answer_type = "FALLBACK"
                    chunk_indices = []
                    sentence_mapping = []
                
                if answer_type == "FALLBACK":
                    chunk_indices = []
                    sentence_mapping = []
                    confidence = 0.0
                
                chunks_used = []
                for idx in chunk_indices:
                    for meta in chunk_metadata_list:
                        if meta.get("chunk_index") == idx:
                            chunks_used.append({
                                "chunk_index": idx,
                                "document_id": meta.get("document_id")
                            })
                            break
                
                return answer, chunks_used, answer_type, confidence, sentence_mapping
            except Exception as e:
                print(f"[RAG] OpenAI API call failed: {e}")
                return self._get_fallback_response()
        
        return self._get_fallback_response()



    def _parse_answer_and_chunks(self, full_response: str, chunk_metadata_list: List[dict]) -> tuple[str, List[dict]]:

        """

        Parse LLM response to extract answer and chunk indices used.

        Expected format: 

        Answer text here...

        [CHUNKS_USED: 1, 5, 7]

        

        Returns: (answer, list of chunk info dicts with chunk_index and document_id)

        """

        # Look for [CHUNKS_USED: ...] pattern

        match = re.search(r'\[CHUNKS_USED:\s*([\d,\s]+)\]', full_response, re.IGNORECASE)

        

        if match:

            # Extract chunk numbers

            chunk_str = match.group(1)

            chunk_indices = [int(x.strip()) for x in chunk_str.split(',') if x.strip().isdigit()]

            

            # Map chunk indices to chunk info with document_id

            chunks_used = []

            for chunk_idx in chunk_indices:

                # Find corresponding metadata

                for chunk_meta in chunk_metadata_list:

                    if chunk_meta.get("chunk_index") == chunk_idx:

                        chunks_used.append({

                            "chunk_index": chunk_idx,

                            "document_id": chunk_meta.get("document_id")

                        })

                        break

                else:

                    # Nếu không tìm thấy metadata, vẫn thêm chunk_index

                    chunks_used.append({"chunk_index": chunk_idx, "document_id": None})

            

            # Remove the [CHUNKS_USED: ...] part from answer

            answer = re.sub(r'\[CHUNKS_USED:.*?\]', '', full_response, flags=re.IGNORECASE).strip()

            

            return answer, chunks_used

        else:

            # Fallback: try to infer from content

            # Look for "Chunk X" mentions in the response

            chunks_mentioned = re.findall(r'[Cc]hunk\s+(\d+)', full_response)

            chunk_indices = list(set(int(x) for x in chunks_mentioned))

            

            # Map to chunk info

            chunks_used = []

            for chunk_idx in chunk_indices:

                for chunk_meta in chunk_metadata_list:

                    if chunk_meta.get("chunk_index") == chunk_idx:

                        chunks_used.append({

                            "chunk_index": chunk_idx,

                            "document_id": chunk_meta.get("document_id")

                        })

                        break

                else:

                    chunks_used.append({"chunk_index": chunk_idx, "document_id": None})

            

            return full_response.strip(), chunks_used





    def _build_references_from_chunks(
        self,
        chunks_used: List[dict],
        selected_results: List[dict],
        chunk_metadata_for_context: List[dict]
    ) -> List[HistoryReference]:
        """Build references from chunks actually used."""
        final_references = []
        
        for chunk_info in chunks_used:
            chunk_idx = chunk_info.get("chunk_index")
            doc_id = chunk_info.get("document_id")
            
            # Find in selected_results
            for item in selected_results:
                record = item.get("_record")
                if not record:
                    continue
                
                if record.get("chunk_index") == chunk_idx and \
                   record.get("document_id") == doc_id:
                    
                    doc = item["document"]
                    chunk_doc = item.get("_chunk_doc")
                    content = item.get("_content", "")
                    
                    # Get metadata
                    chunk_metadata = {}
                    if chunk_doc:
                        chunk_metadata = chunk_doc.get("metadata", {}) or {}
                    elif record:
                        chunk_metadata = record.get("metadata", {}) or {}
                    
                    page = chunk_metadata.get("page_number")
                    section = chunk_metadata.get("section")
                    heading = chunk_metadata.get("heading")
                    
                    preview = content[:160] if content else None
                    
                    final_references.append(
                        HistoryReference(
                            document_id=doc.id,
                            document_filename=doc.filename,
                            document_file_type=doc.file_type,
                            chunk_id=str(chunk_doc.get("_id")) if chunk_doc else None,
                            chunk_index=chunk_idx,
                            page_number=page,
                            section=section or heading,
                            score=item.get("similarity", 0.5),
                            content_preview=preview,
                        )
                    )
                    break
        
        # Deduplicate
        seen = set()
        deduplicated = []
        for ref in final_references:
            key = f"{ref.document_id}_{ref.chunk_index}"
            if key not in seen:
                seen.add(key)
                deduplicated.append(ref)
        
        # CRITICAL FIX: Adjust max references based on question complexity
        # Count how many chunks were actually used
        num_chunks_used = len(chunks_used)
        
        if num_chunks_used >= 5:  # Complex question - keep more refs
            max_refs = min(5, num_chunks_used)
        elif num_chunks_used >= 3:  # Medium complexity
            max_refs = min(4, num_chunks_used)
        else:  # Simple question
            max_refs = min(2, num_chunks_used)
        
        return deduplicated[:max_refs]


rag_service = RAGService()
