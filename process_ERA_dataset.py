import json
import sys
import argparse
import os

def process_files(filename_path, caps_path, output_json, prefix=""):
    with open(filename_path, 'r', encoding='utf-8') as f:
        image_files = [line.strip() for line in f.readlines() if line.strip()]

    with open(caps_path, 'r', encoding='utf-8') as f:
        captions = [line.strip() for line in f.readlines() if line.strip()]

    image_count = len(image_files)
    caption_count = len(captions)

    if image_count == caption_count:
        mode = "1:1 Matching"
    elif caption_count == image_count * 5:
        mode = "1-to-5 Matching"
    else:
        print(f"Error: {image_count} images but {caption_count} captions found.")
        print(f"Expected {image_count * 5} captions for a 1-to-5 match.")
        sys.exit(1)

    print(f"Processing mode: {mode}")


    result = []
    if mode == "1:1 Matching":
        for img, cap in zip(image_files, captions):
            result.append({
                "image_id": img,
                "image": prefix + img,
                "caption": [cap]
            })

    else: # 1-to-5 Matching
        for i, img in enumerate(image_files):
            cap_group = captions[i * 5: i * 5 + 5]
            result.append({
                "image_id": img,
                "image": prefix + img,
                "caption": cap_group
            })

    grouped_result = {}
    for entry in result:
        img_id = entry["image_id"]
        if img_id in grouped_result:
            if isinstance(entry["caption"], list):
                grouped_result[img_id]["caption"].extend(entry["caption"])
            else:
                grouped_result[img_id]["caption"].append(entry["caption"])
        else:
            new_entry = {
                "image_id": entry["image_id"],
                "image": entry["image"],
                "caption": entry["caption"] if isinstance(entry["caption"], list) else [entry["caption"]]
            }
            grouped_result[img_id] = new_entry
    result = list(grouped_result.values())

    with open(output_json, 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=4)
    
    print(f"Successfully created {output_json}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Process image filenames and captions to generate a JSON annotation file."
    )
    parser.add_argument(
        "data_path", 
        type=str, 
        help="Path to the directory containing 'test_filename.txt' and 'test_caps.txt'. Example: VCSR/data/era_precomp"
    )
    parser.add_argument(
        "output_json", 
        type=str, 
        help="Path for the output JSON file. Example: test.json"
    )

    parser.add_argument(
        "--prefix", 
        type=str, 
        default="", 
        help="Optional prefix to add to the image path. Defaults to an empty string."
    )

    args = parser.parse_args()

    filename_path = os.path.join(args.data_path, "test_filename.txt")
    caps_path = os.path.join(args.data_path, "test_caps.txt")

    process_files(
        filename_path=filename_path,
        caps_path=caps_path,
        output_json=args.output_json,
        prefix=args.prefix,
    )