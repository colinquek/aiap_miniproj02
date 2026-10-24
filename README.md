# README
| Attribute | Description |
|---|---|
| booking_id | Unique customer booking ID |
| no_show | If the customer is a No-Show: 0 = Show, 1 = No-Show |
| branch | Hotel branch |
| booking_month | Month the booking was made by the customer |
| arrival_month | Month the customer plan to arrive at the hotel |
| arrival_day | Day date the customer plan to arrive at the hotel |
| checkout_month | Month the customer plan to checkout of the hotel |
| checkout_day | Day date the customer plan to checkout of the hotel |
| country | Nationality of the customer |
| first_time | If it is the first time customer staying in the hotel |
| room | Room type booked by the customer |
| price | Price of the room booked by the customer |
| platform | Platform used to book the room by the customer |
| num_adults | Number of adults staying |
| num_children | Number of children staying |

## Executive Summary
This production machine learning pipeline predicts customer no-shows (`no_show`: `0` = Show, `1` = No-Show) for a hotel chain to optimize booking management and minimize operational losses.

## Repository Structure
```
├── data/                       # Directory for processed data outputs
├── models/                     # Saved model artifacts (.joblib)
│   └── best_no_show_model.joblib
├── src/                        # Production modules
│   ├── config.yaml             # Pipeline parameters & feature lists
│   ├── data_preparation.py     # DataPreparation class
│   └── model_training.py       # ModelTraining class
├── eda.ipynb                   # Interactive EDA & prototyping notebook
├── main.py                     # Pipeline orchestrator
├── requirements.txt            # Dependency configuration
└── README.md                   # Project documentation
```

## How to Run
```bash
# Install dependencies
pip install -r requirements.txt

# Execute full pipeline
python main.py
```
