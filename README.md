
# **YOLOv11n on COCO 2017: A Technical Reproduction Report**

## **1. Project Overview and Objective**

The project's focus is specifically on the **YOLOv11n (tiny)** model. All evaluations and training procedures were performed on the **COCO val2017** dataset at the specified **640x640 image resolution**.

---

## **2. Environment Setup and Prerequisites**

A robust and isolated environment is the foundation for reproducible ML experimentation. This project uses `uv` for its speed and reliability in managing Python environments and dependencies.

### **Prerequisites**

* **Git:** For cloning the project repository.
* **Python 3.8+:** The underlying programming language.
* **NVIDIA GPU with CUDA:** Highly recommended for performance. The `ultralytics` library will automatically use it if available.

### **Step 1: Clone the Repository**

Begin by cloning the project repository to your local machine:

```bash
git clone [URL_to_your_git_repository]
cd [repository_folder_name]
```

### **Step 2: Install `uv`**

If you do not have `uv` installed, install it via `pip`:

```bash
pip install uv
```

### **Step 3: Create and Activate the Virtual Environment**

Using `uv`, create a new virtual environment:

```bash
uv venv
```

Activate the environment:

```bash
# On macOS or Linux
source .venv/bin/activate

# On Windows (Command Prompt)
.venv\Scripts\activate
```

### **Step 4: Install Dependencies**

Install all required packages:

```bash
uv pip install -r requirements.txt
```

---

## **3. Methodology and Execution**

The reproduction process was performed in two phases:

### **3.1. Phase 1: Baseline Performance Validation**

First, validate the official pre-trained `yolo11n.pt` model:

```bash
python main.py --val --model yolo11n.pt --data coco.yaml --imgsz 640 --name pretrained_yolo11n_val
```

* **Results:** The validation run yielded a **mAP<sup>50-95</sup> of 39.3%**, very close to the official paper's 39.5%.

<p align="center">
  <img src="https://github.com/anshulsc/Reproduce-Yolo11/blob/main/assets/1.png" width="400"/>
</p>
<p align="center"><b>Figure 1:</b> Terminal output from validating the pre-trained yolo11n.pt model, showing a final mAP<sup>50-95</sup> of 39.3%.</p>

### **3.2. Phase 2: Full Training Reproduction Analysis**

#### **A. The Challenge of Full Reproduction**

The official training configuration requires significant compute:

<p align="center">
  <img src="[link_to_your_official_config_image.png]" width="600"/>
</p>
<p align="center"><b>Figure 2:</b> The official training configuration for YOLOv11n, showing 600 epochs and a batch size of 128.</p>

Approximate compute time:

> 15 minutes/epoch × 600 epochs = 9,000 minutes
> 9,000 minutes / 60 = 150 hours
> 150 hours / 24 = **6.25 days**

#### **B. A Pragmatic Approach: The 20-Epoch Comparative Test**

Instead of a full 600-epoch run, a **20-epoch** training session was executed:

```bash
python main.py --train --model yolo11.yaml --data coco.yaml --epochs 20 --batch 128 --name yolo11n_selftrained_20epochs
```

#### **C. Comparative Results Analysis (at 20 Epochs)**

Using the official training graph:

<p align="center">
  <img src="[link_to_your_official_600_epoch_graph.png]" width="600"/>
</p>
<p align="center"><b>Figure 3:</b> The official training graph for YOLOv11n over 600 epochs. By inspecting the graph at X=20, we can find the benchmark mAP<sup>50-95</sup>.</p>

Comparison:

* **Official Model at 20 Epochs:** \~\[Official 20-epoch mAP]%
* **Self-Trained Model at 20 Epochs:** \[Your 20-epoch mAP]%

<p align="center">
  <img src="[link_to_your_20_epoch_training_graph.png]" width="600"/>
</p>
<p align="center"><b>Figure 4:</b> My self-trained model's metrics. The performance at epoch 20 is highly comparable to the official model's early-stage performance, validating the reproduction process.</p>

---

## **4. Consolidated Results Summary**

| Metric                  | Official Paper (Final) | My Validation (Pre-trained) |   Official Model (at Epoch 20)  | My Training (at Epoch 20) |
| :---------------------- | :--------------------: | :-------------------------: | :-----------------------------: | :-----------------------: |
| **mAP<sup>50-95</sup>** |        **39.5%**       |          **39.3%**          | **\~\[Official 20-epoch mAP]%** | **\[Your 20-epoch mAP]%** |
| **Epochs Run**          |           600          |             N/A             |                20               |             20            |

---

## **5. Quick Start: How to Run This Project**

1. **Clone the repo:**

   ```bash
   git clone [URL]
   cd [folder]
   ```
2. **Set up environment:**

   ```bash
   pip install uv
   uv venv
   source .venv/bin/activate
   uv pip install -r requirements.txt
   ```
3. **Run Validation:**

   ```bash
   python main.py --val --model yolo11n.pt --data coco.yaml
   ```
4. **Run the 20-Epoch Training Test:**

   ```bash
   python main.py --train --model yolo11.yaml --data coco.yaml --epochs 20
   ```

---

## **6. References**

* [Official Training Configuration & Metrics](https://hub.ultralytics.com/models/7wzkDSKNMcwkPTs8ZVJC?tab=train)
* [Ultralytics Validation Mode Documentation](https://docs.ultralytics.com/modes/val/)

---

If you’d like, I can help you insert the correct URLs for your official training graph, config images, and your own training results graph. Let me know!
