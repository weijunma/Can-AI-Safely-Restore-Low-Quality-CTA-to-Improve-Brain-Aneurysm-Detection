**DOI:** [10.5281/zenodo.19363880](https://doi.org/10.5281/zenodo.19363880)
# Safe AI Restoration of Low-Quality CTA for Brain Aneurysm Detection

A student research project on whether AI can safely restore degraded CT angiography (CTA) to support brain aneurysm detection without introducing harmful image changes.

This work focuses not on making scans simply look clearer, but on making them more useful for aneurysm-related interpretation while preserving anatomy and keeping edits cautious.

## Project Summary

Intracranial aneurysms can lead to severe disability or death if they rupture, so early and accurate detection is important. CTA is widely used because it is fast, accessible, and practical in clinical settings. However, real CTA scans are often affected by noise, blur, motion artifact, low-dose acquisition, and reconstruction artifact, which can obscure subtle vascular details.

This project asks a safety-centered question: can AI restore low-quality CTA in a way that improves aneurysm-relevant signal while minimizing harmful modifications?

Because perfectly paired degraded and clean clinical scans are rarely available, this project uses relatively clean CT/CTA images and applies realistic synthetic degradations to create training pairs. The model is trained to make limited, conservative corrections to the original scan rather than freely redrawing the image.

## Motivation

Many medical image enhancement methods report improvements in metrics such as PSNR or SSIM. However, in medical imaging, a better-looking image is not automatically a safer or more clinically useful one. For brain aneurysm detection, tiny vessel boundaries and subtle structural details matter. Over-editing can remove important anatomy or introduce misleading features.

The central goal of this project is therefore not aggressive enhancement, but cautious restoration.

## Method

This project uses a 2.5D restoration approach. The model reads the previous slice, current slice, and next slice, but restores only the center slice. This allows the method to use local anatomical context without the added complexity of full 3D processing.

To simulate realistic low-quality CTA conditions, the training pipeline includes degradations such as:
- blur
- motion artifact
- Poisson-Gaussian noise
- ring or band artifact
- edge-streak or reconstruction-style artifact

The model is designed to perform bounded corrections on top of the original image, with the goal of preserving anatomy and reducing unnecessary edits.

## Evaluation

The project evaluates restoration performance using several complementary strategies:

### 1. Clinical Rescue Matrix
Measures whether restored images improve aneurysm-relevant downstream signal compared with degraded inputs and baseline methods.

### 2. Monte Carlo Stability Testing
Repeats restoration across multiple random degradation seeds to test whether the method stays net positive under stochastic variation.

### 3. Anatomical Overlap Analysis
Examines where meaningful edits occur, helping determine whether the model is acting on relevant anatomical regions rather than modifying the image indiscriminately.

### 4. External Low-Dose Testing
Tests whether the method can generalize beyond synthetic degradations to external low-dose CT data.

## Preliminary Results

Early results include:
- Mean target gain: **0.0635**
- PSNR: **37.51 dB**
- Iatrogenic risk rate: **4.0%**
- Higher target gain than a Gaussian baseline in **64%** of cases
- Positive outcome in **85.4%** of Monte Carlo runs
- **No stably negative cases** observed in Monte Carlo testing

These results suggest that the method may improve degraded CTA in a cautious and clinically meaningful way, rather than only improving visual appearance.

## Repository Structure

This repository may contain:
- `*.ipynb` Kaggle notebooks
- helper Python scripts
- small metadata files such as UID lists
- selected result figures
- documentation and project notes

Large external datasets and model assets may not be included.

## Installation

This project was developed in a Kaggle/Python environment.

Typical dependencies may include:

```txt
numpy
pandas
torch
opencv-python
pydicom
matplotlib
scikit-image
