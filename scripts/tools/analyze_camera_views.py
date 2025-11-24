
import os
import glob
import base64
import requests

# Configuration
CALIB_DIR = "/home/kimate/open_arm_10Things/IsaacLab/calibration_images"
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
            "content": "You are a robotics computer vision expert. You are calibrating a wrist-mounted camera on a robotic arm. The goal is to have a clear view of the gripper fingers and the workspace immediately in front of them (where an object would be grasped)."
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": "I have generated a set of camera views with different pitch angles. Please analyze these images. For each image, tell me:\n1. Can you see the gripper fingers?\n2. Can you see the target area (the cube)?\n3. Is the camera tilted too far down (seeing only robot parts) or too far up (seeing over the workspace)?\n\nFinally, recommend which pitch angle provides the best trade-off."
                }
            ]
        }
    ]

    for img_path in image_paths:
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

    print(f"Sending {len(image_paths)} images to GPT-4o for analysis...")
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
    # Find all right wrist images
    images = sorted(glob.glob(os.path.join(CALIB_DIR, "*_right.png")))
    
    if not images:
        print(f"No images found in {CALIB_DIR}. Did you run calibrate_wrist_cameras.py?")
        return

    print(f"Found {len(images)} images.")
    analyze_images(images)

if __name__ == "__main__":
    main()
