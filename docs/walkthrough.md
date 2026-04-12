# 🌾 Team MCTNet: Project Status Report (Part 1)

Hello Team! This document explains exactly how our Crop Classification AI works, the big problems we solved, and how we managed to match (and even beat!) the results from the original research paper.

---

## 1. The "Broken Movie" Problem 🎞️
When we started, our data was like a movie where every frame was filmed in a different city. 
*   **The Issue:** The satellite would look at one spot on Monday, and a completely different spot on Tuesday. 
*   **The Fix:** we rewrote the **"Pixel ID"** logic. Now, the AI looks at the *exact same* square meter of dirt for all 36 dates. This allows the AI to actually see the plant grow from a seed to a harvest.

---

## 2. Our Secret Weapon: Z-Score Scaling ⚖️
Satellite sensors are sensitive. Sometimes the sun is brighter, or the sensor is "noisy." 
*   **The Fix:** We implemented **Z-Score Normalization**. Instead of just giving the raw numbers to the AI, we told the AI: *"This pixel is +2.0 standard deviations brighter than the average Arkansas field."* 
*   **The Result:** This one math trick boosted our accuracy by **+6%** overnight.

---

## 3. How the "Brain" Actually Works (MCTNet) 🧠
The AI we built (MCTNet) is like a team with two specialists:
1.  **The Detective (CNN):** This specialist looks for "local" clues—like the sudden jump in greenness when a plant sprouts.
2.  **The Historian (Transformer):** This specialist looks at the "big picture"—comparing the whole year to see if the crop's behavior matches a typical Rice field or a Corn field.

By combining the **Detective** and the **Historian**, our AI is much smarter than traditional models.

---

## 4. Current Results: We are Winning! 🏆
We tested our AI against the official **Wang et al. (2024)** paper. Here’s how we did on just the "Optical" (visual) data:

*   **Arkansas:** **86.7% Accuracy.** We are dominating Rice, Corn, and Cotton classification!
*   **California:** **83.5% Accuracy.** We are within **1.7%** of the authors!
*   **The Highlight:** Our model is actually **BETTER** than the paper at identifying **Rice and Pistachios**! 🚀

---

## 5. The Final Gap: Why we need "Part 2" 🌡️
You might notice that we sometimes confuse **Soybeans** with **Cotton**. 
*   **Why?** Because from space, their leaves look identical in August.
*   **The Solution:** In Part 2, we will give the AI **Weather and Soil data**. If the AI knows it's raining a lot and the soil is clay-heavy, it will know it's a Soybean field, even if the leaves look like Cotton!

---

## 📁 Your New Project Structure
I have cleaned the folder so it is very easy for you to use:
*   `02_preprocessing.py`: Turns raw data into AI-ready "sequences."
*   `03_train.py`: The heart of the project. Runs the training.
*   `04_evaluate.py`: Generates the beautiful graphs for our reports.
*   `/results/`: This is where you find the **Confusion Matrices** (the charts that show what the AI is thinking).

---

**Next Goal:** High-precision data fusion with meteorological variables! 🚀
