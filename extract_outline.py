import os
import json
import fitz  # PyMuPDF
import re
import numpy as np
import joblib

# ---------------------- PDF Feature Extraction ----------------------

def extract_pdf_features(pdf_path):
    result = []
    doc = fitz.open(pdf_path)

    for page_num, page in enumerate(doc):
        spans = []
        for block in page.get_text("dict")["blocks"]:
            if "lines" not in block:
                continue
            for line in block["lines"]:
                for span in line["spans"]:
                    spans.append({
                        "text": span["text"],
                        "size": round(span["size"], 2),
                        "flags": span["flags"],
                        "font": span["font"],
                        "bbox": span["bbox"],
                        "origin": (span["bbox"][0], span["bbox"][1])
                    })

        spans.sort(key=lambda s: (round(s["origin"][1]), s["origin"][0]))

        # Merge spans into lines
        lines, current_line, current_y = [], [], None
        y_threshold = 2

        for span in spans:
            y = round(span["origin"][1])
            if current_y is None or abs(y - current_y) <= y_threshold:
                current_line.append(span)
                current_y = y
            else:
                lines.append(merge_line(current_line))
                current_line = [span]
                current_y = y
        if current_line:
            lines.append(merge_line(current_line))

        # Merge lines into paragraphs
        paragraphs = []
        if lines:
            current_para = lines[0].copy()
            for i in range(1, len(lines)):
                same_font = lines[i]["font"] == lines[i - 1]["font"]
                same_size = lines[i]["size"] == lines[i - 1]["size"]
                if same_font and same_size:
                    current_para["text"] += " " + lines[i]["text"]
                    current_para["bbox"][3] = max(current_para["bbox"][3], lines[i]["bbox"][3])
                else:
                    paragraphs.append(current_para)
                    current_para = lines[i].copy()
            paragraphs.append(current_para)

        result.append({
            "page_number": page_num,
            "width": page.rect.width,
            "height": page.rect.height,
            "text_blocks": paragraphs
        })

    return result

def merge_line(spans):
    if not spans:
        return {}
    spans.sort(key=lambda s: s["origin"][0])
    full_text = "".join(span["text"] for span in spans).strip()

    # Clean trailing digits and excessive dots
    full_text = re.sub(r'\d+$', '', full_text).rstrip()
    full_text = re.sub(r'[.-]{4,}$', '', full_text).rstrip()

    return {
        "text": full_text,
        "size": round(max(s["size"] for s in spans), 2),
        "flags": spans[0]["flags"],
        "font": spans[0]["font"],
        "bbox": [
            spans[0]["bbox"][0],
            spans[0]["bbox"][1],
            spans[-1]["bbox"][2],
            max(s["bbox"][3] for s in spans)
        ]
    }

# ---------------------- Classification ----------------------

def classify(sampled_blocks, basename, save_dir):
    # with open(json_path, encoding='utf8') as f:
    #     sampled_blocks = json.load(f)

    clf = joblib.load("best_model.pkl")
    # text_model = joblib.load("sentence_transformer.pkl")
    font_enc = joblib.load("font_encoder.pkl")
    layout_scaler = joblib.load("layout_scaler.pkl")
    label_enc = joblib.load("label_encoder.pkl")

    # texts = [b["text"] for b in sampled_blocks]
    sizes = [b["size"] for b in sampled_blocks]
    bboxes = [b["bbox"] for b in sampled_blocks]
    fonts = [b["font"] for b in sampled_blocks]

    # Encode text
    # text_embeddings = text_model.encode(texts, batch_size=32, show_progress_bar=True)

    # Encode and normalize layout features
    font_features = font_enc.transform(np.array(fonts).reshape(-1, 1)).toarray()
    layout_raw = np.hstack([np.array(sizes).reshape(-1, 1), np.array(bboxes), font_features])
    layout_features = layout_scaler.transform(layout_raw)

    # Final feature vector
    X = np.hstack([layout_features])

    # Predict and decode
    y_pred = clf.predict(X)
    y_pred_labels = label_enc.inverse_transform(y_pred)

    for block, pred in zip(sampled_blocks, y_pred_labels):
        block["predicted_level"] = pred

    title_text = next((b["text"] for b in sampled_blocks if b["predicted_level"] == "title"), "")

    outline = [{
        "level": b["predicted_level"],
        "text": b["text"],
        "page": b["page"]
    } for b in sampled_blocks if b["predicted_level"] not in ["body", "title"]]

    outline.sort(key=lambda x: x["page"])

    final_output = {
        "title": title_text,
        "outline": outline
    }

    os.makedirs(save_dir, exist_ok=True)
    fname = basename + "_outline.json"
    save_path = os.path.join(save_dir, fname)

    with open(save_path, "w", encoding="utf8") as f:
        json.dump(final_output, f, indent=2, ensure_ascii=False)

    # print(f" Saved outline: {save_path}")


# ---------------------- Main Batch Pipeline ----------------------

def process_all_pdfs(input_dir="input", json_dir="jsondata", output_dir="output"):
    os.makedirs(json_dir, exist_ok=True)

    for fname in os.listdir(input_dir):
        if not fname.endswith(".pdf"):
            continue
        pdf_path = os.path.join(input_dir, fname)
        base_name = os.path.splitext(fname)[0]

        # Extract PDF features
        print(f"📄 Extracting from {pdf_path}")
        features = extract_pdf_features(pdf_path)

        # with open(f"jsondata/{base_name}feature.json", "w", encoding="utf8") as f:
        #     json.dump(features, f, indent=2, ensure_ascii=False)

        # Flatten features
        blocks = []
        for page in features:
            for b in page["text_blocks"]:
                b["page"] = page["page_number"]
                blocks.append(b)

        # Save JSON
        # with open(json_path, "w", encoding="utf8") as f:
        #     json.dump(blocks, f, indent=2, ensure_ascii=False)

        # Run classification
        classify(blocks,base_name, save_dir=output_dir)

# ---------------------- Run All ----------------------

if __name__ == "__main__":
    process_all_pdfs()
