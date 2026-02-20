[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/2iEyTCx6)
# Lab 1: Campaign Finance & Ideology Estimation with PCA

In this lab you will explore campaign finance data from the 2024 election cycle and use **Principal Component Analysis (PCA)** to estimate the political ideology of donors and recipients.

## Getting Started with GitHub Codespaces

The easiest way to complete this lab is using GitHub Codespaces, which runs everything in your browser.

### Step 1: Open in Codespaces

Click the green **Code** button above, then select **Open with Codespaces** > **New codespace**.

### Step 2: Wait for Setup

The codespace will:
1. Build the development environment (1-2 minutes)
2. Install required packages
3. **Automatically open Marimo** in a new browser tab

If Marimo doesn't open automatically, you can run:
```bash
uv run marimo edit notebooks/lab01/lab01.py --host 0.0.0.0 --port 2718
```

### Step 3: Complete the Notebook

In the Marimo interface, work through the exercises:
- **Part A**: Explore donation patterns by party and industry
- **Part B**: Apply PCA to estimate ideology scores

### Step 4: Submit Your Work

Once you've completed the notebook, commit and push your changes:

**Option A: Using the terminal**
```bash
git add notebooks/lab01/lab01.py
git commit -m "Complete Lab 1"
git push
```

**Option B: Using VS Code**
1. Click the Source Control icon in the left sidebar
2. Stage your changes (+ button)
3. Enter a commit message
4. Click "Commit" then "Sync Changes"

## Running Locally (Alternative)

If you prefer to run locally instead of using Codespaces:

1. Install [uv](https://docs.astral.sh/uv/getting-started/installation/)
2. Clone this repository
3. Run:
   ```bash
   uv sync
   uv run marimo edit notebooks/lab01/lab01.py
   ```

## Troubleshooting

**Marimo tab didn't open?**
- Check the "Ports" tab in VS Code and click on port 2718
- Or run the marimo command manually (see Step 2)

**Package errors?**
- Try running `uv sync` in the terminal

**Can't push changes?**
- Make sure you accepted the GitHub Classroom assignment first
- Check that you're signed into GitHub in the codespace

## Need Help?

If you encounter issues, post on the course discussion board or come to office hours.
