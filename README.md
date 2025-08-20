
# Movie Recommendation & Insights System

This repository contains:
- A synthetic dataset: `movies_synthetic_dataset.csv`
- A ready-to-run Jupyter Notebook: `PalakPandey_BatchX_MovieProject.ipynb`

## How to Run
1. Open the notebook in Jupyter or VS Code.
2. Install dependencies if needed:
   ```
   pip install pandas numpy scikit-learn matplotlib seaborn
   ```
3. Run all cells.

## File Naming
Rename the notebook to match your submission format: **YourName_Batch_MovieProject.ipynb**.

## Replace Dataset (Optional)
You can replace the synthetic CSV with Kaggle's IMDB 5000 Movie Dataset. Map/rename columns to:
`Title, Genre, Runtime, Budget_M, Votes, Rating, Revenue_M, Hit`

Define **Hit** as either:
- `Revenue_M > Budget_M`, or
- `Rating > 7.5`

Then re-run the notebook.
