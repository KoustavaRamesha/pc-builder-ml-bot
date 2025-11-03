# PC Building Assistant Bot

A machine learning-based PC building assistant that helps users with computer hardware recommendations and build advice.

## Features

- **No API Calls**: Fully self-contained machine learning model
- **Component Recommendations**: Suggests CPUs, GPUs, RAM, motherboards, PSUs, etc. with pricing
- **Complete Build Guidance**: Provides full PC configurations with motherboard and PSU included
- **Cost Estimation**: Shows individual component prices and total build cost
- **Model Persistence**: Saves and loads trained models
- **Interactive Interface**: Command-line interface for real-time assistance

## Requirements

- Python 3.7+
- scikit-learn
- pandas
- numpy
- joblib

## Installation

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Run the console bot:
   ```bash
   python src/pc_bot.py
   ```

3. Or run the web app:
   ```bash
   python src/web_app.py
   ```
   Then open http://localhost:5000 in your browser

## Usage

1. The bot will automatically train a model using the sample data in `data/pc_building_data.csv`
2. Ask questions about PC components or builds
3. Type 'quit' to exit

## Project Structure

```
ai ml bot/
├── data/
│   └── pc_building_data.csv  # Training data
├── src/
│   ├── pc_bot.py             # Main bot script
│   └── test_bot.py           # Test script
├── models/                   # Saved models (created after training)
├── requirements.txt          # Python dependencies
└── README.md
```

## How It Works

1. **Query Classification**: Uses ML to categorize user questions (CPU, GPU, RAM, etc.)
2. **Recommendation Engine**: Provides specific hardware recommendations based on category
3. **Text Processing**: Converts queries to numerical features using TF-IDF vectorization
4. **Model Training**: Trains a Logistic Regression classifier on query-category pairs
5. **Interactive Responses**: Gives personalized PC building advice

## Supported Categories

- **CPU**: 11 options from ultra-budget ($109) to professional ($799)
  - AMD Ryzen 5 5600G, Ryzen 5 5600, Ryzen 5 7600, Ryzen 5 7600X, Ryzen 7 7700X, Ryzen 9 7950X
  - Intel Core i3-12100F, i5-12400F, i5-13600K, i7-13700K, i9-13900K
- **GPU**: 12 options from ultra-budget ($149) to professional ($1699)
  - NVIDIA GTX 1650, RTX 4060, 4060 Ti, 4070, 4070 Ti, 4080, 4090
  - AMD RX 6600, 7600, 7800 XT, 7900 XT
- **RAM**: 7 options from 8GB to 64GB ($29-$299)
  - DDR4 and DDR5 compatibility options
- **Motherboard**: 11 options from budget to enthusiast ($79-$599)
  - ASRock B450M PRO4, ASRock B660M-HDV, MSI PRO B660M-A
  - MSI MAG B650 Tomahawk, ASUS ROG Strix B650-A Gaming
  - MSI MAG Z790 Tomahawk, ASUS ROG Strix Z790-A Gaming
  - ASUS ProArt B650-Creator, ASUS ProArt Z790-Creator
  - ASUS ROG Crosshair X670E Hero, ASUS ROG Maximus Z790 Hero
- **PSU**: 7 options from 450W to 1200W ($49-$299)
  - Various wattages for different build requirements
- **Case**: PC case suggestions ($69-$149)
- **Cooler**: CPU cooling solutions ($39-$149)
- **Storage**: 9 options from budget SSD/HDD to high-end SSD ($39-$179)
  - SATA SSD, NVMe SSD, and HDD options
- **Budget Builds**: Complete builds (~$650-$800)
- **Gaming Builds**: High-performance gaming PC configurations (~$1300-$2200)
- **Office Builds**: Productivity workstation setups (~$650)
- **Workstation**: Content creation and professional builds (~$2800)

## Special Features

- **Interactive PC Building**: Ask "help me build a pc" to get personalized recommendations
- **Component Compatibility**: All components are compatible (CPU/motherboard socket, RAM type, etc.)
- **Educational Responses**: Ask "what is a CPU?" or "why do I need RAM?" for explanations
- **Fun Facts**: Each component explanation includes an interesting computer history fact
- **Total Cost Calculation**: Complete builds show individual prices and grand total

## Customization

- Add more training data to `data/pc_building_data.csv`
- Update recommendations in the `recommendations` dictionary
- Modify the model in `src/pc_bot.py` (e.g., use different algorithms)
- Adjust vectorizer parameters for different text processing needs
