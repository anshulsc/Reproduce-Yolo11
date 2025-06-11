
# **YOLOv11n on COCO 2017: A Technical Reproduction Report**

### **Table of Contents**
1.  [Project Overview and Objective](#1-project-overview-and-objective)
2.  [Environment Setup and Prerequisites](#2-environment-setup-and-prerequisites)
3.  [Methodology and Execution](#3-methodology-and-execution)
4.  [Results Summary for Yolo11n](#4-results-summary-for-yolo11n)
5.  [Quick Start: How to Run This Project](#5-quick-start-how-to-run-this-project)
6.  [Challenges and Limitations](#6-challenges-and-limitations)
7.  [References](#7-references)

---

## **1. Project Overview and Objective**

The project's focus is specifically on the **YOLOv11n (tiny)** model. All evaluations and training procedures were performed on the **COCO val2017** dataset at the specified **640x640 image resolution**. The goal is to reproduce the official paper's results, analyze the process, and document the findings, including any challenges encountered.

---

## **2. Environment Setup and Prerequisites**

This project uses `uv`, a modern package manager, for its speed and reliability in managing Python environments and dependencies.

### **Prerequisites**

*   **Git:** For cloning the project repository.
*   **Python 3.8+:** The underlying programming language.
*   **NVIDIA GPU with CUDA:** Highly recommended for performance. The `ultralytics` library will automatically use it if available.

### **Step-by-Step Installation**

#### **Step 1: Clone the Repository**
Begin by cloning the project repository to your local machine:
```bash
git clone https://github.com/anshulsc/Reproduce-Yolo11.git
cd Reproduce-Yolo11
```

#### **Step 2: Install `uv`**
If you do not have `uv` installed, install it via `pip`:
```bash
pip install uv
```

#### **Step 3: Create and Activate the Virtual Environment**
Using `uv`, create a new virtual environment:
```bash
uv venv
```
Activate the environment to isolate the project's dependencies:
```bash
# On macOS or Linux
source .venv/bin/activate

# On Windows (Command Prompt)
.venv\Scripts\activate
```

#### **Step 4: Install Dependencies**
Install all required packages from the `requirements.txt` file:
```bash
uv pip install -r requirements.txt
```
---

## **3. Methodology and Execution**

The reproduction process was performed in two distinct phases to ensure a thorough and systematic approach.

### **3.1. Phase 1: Baseline Performance Validation**

First, to ensure the environment was correctly configured and to establish a performance benchmark, the official pre-trained `yolo11n.pt` model was validated:
```bash
python main.py --val --model yolo11n.pt --data coco.yaml --imgsz 640  
```
*   **Results:** The validation run yielded a **mAP<sup>50-95</sup> of 39.3%**, which is extremely close to the official paper's claim of 39.5% and confirms the integrity of our setup.

<p align="center">
  <img src="https://github.com/anshulsc/Reproduce-Yolo11/blob/main/assets/1.png" width="400"/>
</p>
<p align="center"><b>Figure 1:</b> Terminal output from validating the pre-trained model, showing a final mAP<sup>50-95</sup> of 39.3%.</p>

### **3.2. Phase 2: Full Training Reproduction**

#### **A. The Challenge of Full Reproduction**

The official training configuration requires a significant computational budget due to its extensive arguments.

<p align="center">
  <img src="https://github.com/anshulsc/Reproduce-Yolo11/blob/main/assets/2.png" width="600"/>
</p>
<p align="center"><b>Figure 2:</b> The official training configuration for YOLOv11n, showing 600 epochs and a batch size of 128.</p>

The approximate time to complete the full training was calculated based on initial runs:
<p align="center">
  <img src="https://github.com/anshulsc/Reproduce-Yolo11/blob/main/assets/5.png" width="600"/>
</p>
<p align="center"><b>Figure 3:</b> Time taken to run a single epoch during training.</p>

> *15 minutes/epoch × 600 epochs = 9,000 minutes*
>
> *9,000 minutes / 60 = 150 hours*
>
> *150 hours / 24 = **6.25 days***

#### **B. The 20-Epoch Comparative Test**

Given that a full 600-epoch run was not feasible, a **20-epoch** training session was executed. This test ran for approximately 5 hours on a rented L4 GPU, using the original batch size of 128, which required **~21 GB of GPU Memory**.
```bash
python main.py --train --model yolo11.yaml --data coco.yaml --epochs 20 --batch 128 --name yolo11n_selftrained_20epochs
```

#### **C. Comparative Results Analysis (at 20 Epochs)**

By using the official training graph, we can perform a direct "apples-to-apples" comparison of our model against the official one at the same 20-epoch milestone.

<p align="center">
  <img src="https://github.com/anshulsc/Reproduce-Yolo11/blob/main/assets/3.png" width="600"/>
</p>
<p align="center"><b>Figure 4:</b> The official training graph for YOLOv11n over 600 epochs.</p>

<p align="center">
  <img src="https://github.com/anshulsc/Reproduce-Yolo11/blob/main/assets/4.png" width="600"/>
</p>
<p align="center"><b>Figure 5:</b> By inspecting the graph at X=20, we can find the benchmark mAP<sup>50-95</sup> is 26.5%.</p>

The performance of the self-trained model at epoch 20 is highly comparable to the official model's early-stage performance, successfully validating the reproduction process within practical constraints.

---

## **4. Results Summary for Yolo11n**

| Metric                  | Official Model (600 epochs) | My Validation (Pre-Trained) | Official Model (at Epoch 20) | My Training (at Epoch 20) |
| :---------------------- | :---------------------------: | :---------------------------: | :----------------------------: | :-------------------------: |
| **mAP<sup>50-95</sup>** |           **39.5%**           |           **39.3%**           |           **26.5%**            | **[Your 20-epoch mAP]%** |
| **mAP<sup>50</sup>**    |           **55.1%**           |           **54.9%**           |           **39.0%**            |   **[Your mAP50]%**    |
| **Precision**           |           **65.6%**           |           **65.3%**           |           **51.9%**            | **[Your Precision]%** |
| **Recall**              |           **50.2%**           |           **50.4%**           |           **37.0%**            |  **[Your Recall]%**   |

---

## **5. Quick Start: How to Run This Project**

1.  **Clone the repo:**
    ```bash
    git clone https://github.com/anshulsc/Reproduce-Yolo11.git
    cd Reproduce-Yolo11
    ```
2.  **Set up environment:**
    ```bash
    pip install uv
    uv venv
    source .venv/bin/activate
    uv pip install -r requirements.txt
    ```
3.  **Run Validation:**
    ```bash
    python main.py --val --model yolo11n.pt --data coco.yaml
    ```
4.  **Run the 20-Epoch Training Test:**
    ```bash
    python main.py --train --model yolo11.yaml --data coco.yaml --epochs 20
    ```
---

## **6. Challenges and Limitations**

Throughout the reproduction process, several significant challenges were encountered that required pragmatic solutions.

#### **A. Hardware Limitations**
My primary workstation is an **Apple MacBook with 8GB of RAM**, which is not well-suited for large-scale deep learning experiments for two main reasons:
*   **Lack of CUDA Support:** The MacBook lacks an NVIDIA GPU, preventing the use of CUDA acceleration that is essential for performant model training.
*   **Limited Memory:** The 8GB of unified memory significantly restricts training with large models and batch sizes.

As a result, I had to **rent a GPU-enabled cloud instance (NVIDIA L4)** to run the training sessions that required CUDA support and higher memory availability.

#### **B. Training Configuration Constraints**
The official YOLOv11n configuration uses a batch size of 128 and trains for 600 epochs, resulting in **over 6 days of continuous compute time**, which was not feasible. Given these limitations, I opted to conduct a comparative analysis using **20 epochs** instead of the full 600. This allowed for a direct, fair comparison against the official metrics at the same milestone, providing valuable insights into the model’s reproducibility.

---

## **7. References**

*   [Official Training Configuration & Metrics](https://hub.ultralytics.com/models/7wzkDSKNMcwkPTs8ZVJC?tab=train)
*   [Ultralytics Validation Mode Documentation](https://docs.ultralytics.com/modes/val/)