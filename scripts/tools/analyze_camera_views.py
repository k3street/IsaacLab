
import os
import glob
import base64
import requests

# Configuration
CALIB_DIR = "/home/kimate/open_arm_10Things/IsaacLab/output_images"
API_KEY = os.environ.get("OPENAI_API_KEY")

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def analyze_images(image_paths):
    if not API_KEY:
        print("Error: OPENAI_API_KEY environment variable not set.")
        return

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {API_KEY}"
    }

    messages = [
        {
            "role": "system",
            "content": "You are a robotics expert analyzing a robot arm's behavior. The user reports the robot is 'reaching both arms as far apart from each other as possible' instead of reaching for the cube. Analyze the images to confirm this behavior and look for signs of joint limit issues or incorrect mapping (e.g. arms crossed, twisted, or moving in opposite directions)."
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": "I have generated a set of camera views from an evaluation run. Please analyze these images. For each image, tell me:\n1. Where are the arms relative to the cube? (Reaching towards, reaching away, far apart?)\n2. Does the robot look like it is in a valid pose or a weird/broken pose?\n3. Are the grippers open or closed?\n\nFinally, summarize if the behavior matches 'arms reaching apart' and hypothesize why (e.g. coordinate frame mismatch, sign inversion)."
                }
            ]
        }
    ]

    # Limit to first 5 and last 5 images to save tokens/time
    if len(image_paths) > 10:
        selected_images = image_paths[:5] + image_paths[-5:]
    else:
        selected_images = image_paths

    for img_path in selected_images:
        filename = os.path.basename(img_path)
        base64_image = encode_image(img_path)
        messages[1]["content"].append({
            "type": "text",
            "text": f"Image: {filename}"
        })
        messages[1]["content"].append({
            "type": "image_url",
            "image_url": {
                "url": f"data:image/png;base64,{base64_image}"
            }
        })

    payload = {
        "model": "gpt-4o",
        "messages": messages,
        "max_tokens": 1000
    }

    print(f"Sending {len(selected_images)} images to GPT-4o for analysis...")
    try:
        response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload, timeout=60)
    except requests.exceptions.RequestException as e:
        print(f"Request failed: {e}")
        return

    if response.status_code == 200:
        print("\n--- Analysis Result ---\n")
        print(response.json()['choices'][0]['message']['content'])
    else:
        print(f"Error: {response.status_code} - {response.text}")

def main():
    # Find overhead images
    images = sorted(glob.glob(os.path.join(CALIB_DIR, "ep0_step*_overhead_rgb.png")))
    
    if not images:
        print(f"No images found in {CALIB_DIR}. Did you run evaluate_groot.py?")
        return

    print(f"Found {len(images)} images.")
    analyze_images(images)

if __name__ == "__main__":
    main()
