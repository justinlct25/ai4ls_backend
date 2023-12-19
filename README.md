# This is the backend for the AI4LS Hackathon

## Introductions
AI for Life Sciences(2023) is a challenge series connecting AI enthusiasts with organizations to apply AI in various life sciences domains like biology, genetics, and ecology. Participants work on real-world problems, using AI to create innovative solutions.

## About this Project
  - [Demo Website](http://13.213.141.140/) 
  - [Frontend](https://github.com/bobotangpy/AI4SL-Frontend)
  - [Backend](https://github.com/justinlctstudy96/ai4ls_backend/tree/main)
  - [Data tools](https://github.com/morganluuuu/AI4LS/tree/main)
    
## Steps to run the backend:
- git clone https://github.com/justinlctstudy96/ai4ls_backend.git
- python -m venv venv
- source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
- pip install -r requirements.txt
- python3 backend.py

## APIs: 
### GET API 1: /soil_attribute_units
```
{
    "CaCO3": "g/kg",
    "EC": "mS/m",
    "K": "mg/kg",
    "N": "g/kg",
    "OC": "g/kg",
    "P": "mg/kg",
    "pH_H2O": "-"
}
```

### GET API 2: /soil_attribute_standards
Output:
```
{
    "CaCO3": {
        "max": 10.0,
        "min": 0.0
    },
    "K": {
        "max": 1500.0,
        "min": 121.0
    },
    "N": {
        "min": 1.5
    },
    "OC": {
        "max": 20.0,
        "min": 3.0
    },
    "P": {
        "max": 140.0,
        "min": 16.0
    },
    "pH_H2O": {
        "max": 8.5,
        "min": 5.5
    }
}
```

### GET API3: /land_use_and_cover_classes
Output: 
```
{
    "land_cover_classes": [
        "Artificial land",
        "Bareland",
        "Cropland",
        "Grassland",
        "Shrubland",
        "Water",
        "Woodland"
    ],
    "land_use_classes": [
        "Agriculture (excluding fallow land and kitchen gardens)",
        "Amenities, museum, leisure (e.g. parks, botanical gardens)",
        "Commerce",
        "Construction",
        "Electricity, gas and thermal power distribution",
        "Energy production",
        "Fallow land",
        "Forestry",
        "Kitchen gardens",
        "Other abandoned areas",
        "Other primary production",
        "Protection infrastructures",
        "Railway transport",
        "Residential",
        "Road transport",
        "Semi-natural and natural areas not in use",
        "Sport"
    ]
}
```
### POST API 1 - /land_use_and_cover_for_predictions
Body:
```

{
    "LU1_Desc": ["Agriculture (excluding fallow land and kitchen gardens)"],
    "LC0_Desc": ["Bareland"]
}
```

Output:

```
{
    "all_attributes": {
        "model_accuracy": {
            "BD 0-10": {
                "MAE": 0.23458101712492468
            },
            "BD 0-20": {
                "MAE": 0.22601517462837817
            },
            "BD 10-20": {
                "MAE": 0.2393407523411137
            },
            "CaCO3": {
                "MAE": 59.10032392278259
            },
            "Clay": {
                "MAE": 8.333875509084493
            },
            "Coarse": {
                "MAE": 10.563049454807516
            },
            "EC": {
                "MAE": 10.027024656770392
            },
            "K": {
                "MAE": 114.08801513112535
            },
            "N": {
                "MAE": 1.7725339395210393
            },
            "OC": {
                "MAE": 31.865389869932347
            },
            "P": {
                "MAE": 17.794391721820677
            },
            "Sand": {
                "MAE": 15.9717201331519
            },
            "Silt": {
                "MAE": 10.09256461275031
            },
            "pH_H2O": {
                "MAE": 0.8214400221202984
            }
        },
        "model_info": "Support Vector Regression (SVR) models",
        "result": {
            "BD 0-10": {
                "value": 1.137205843332808
            },
            "BD 0-20": {
                "value": 1.2700000762045593
            },
            "BD 10-20": {
                "value": 1.171359979501251
            },
            "CaCO3": {
                "out_of_standard": false,
                "value": -1.2585212768809448
            },
            "Clay": {
                "value": 29.122465756695682
            },
            "Coarse": {
                "value": 16.243783149646795
            },
            "EC": {
                "value": 14.880585360502469
            },
            "K": {
                "out_of_standard": true,
                "value": 210.22555027920896
            },
            "N": {
                "out_of_standard": true,
                "value": 2.241776370468229
            },
            "OC": {
                "out_of_standard": true,
                "value": 18.182633090126757
            },
            "P": {
                "out_of_standard": true,
                "value": 28.19565970702648
            },
            "Sand": {
                "value": 29.399125131169555
            },
            "Silt": {
                "value": 36.7094573758442
            },
            "pH_H2O": {
                "out_of_standard": true,
                "value": 6.371554408586461
            }
        }
    }
}
```

### POST API1:  /chem_attributes_for_predictions
Body: 
```

{
    "pH_H2O": 4.85,
    "EC": 12.53,
    "OC": 47.5,
    "CaCO3": 1,
    "P": 12.3,
    "N": 3.1,
    "K": 114.8
}
```

Output:
```
{
    "erosion_probability": {
        "model_accuracy": {
            "ROC-AUC": 0.671173600810453
        },
        "model_info": "Support Vector Classification (SVC) model",
        "result": {
            "probability": 0.03
        }
    },
    "is_managed": {
        "model_accuracy": {
            "correct_ratio": 0.8546220700553068
        },
        "model_info": "Support Vector Classification (SVC) model",
        "result": {
            "prediction": 0,
            "probability": 0.7976676714999704
        }
    },
    "phy_attributes_bulk_density": {
        "model_accuracy": {
            "BD 0-10": {
                "MAE": 0.17625830737216408
            },
            "BD 0-20": {
                "MAE": 0.1650467159513966
            },
            "BD 10-20": {
                "MAE": 0.18264405934946643
            }
        },
        "model_info": "Support Vector Regression (SVR) models",
        "result": {
            "BD 0-10": {
                "value": 0.8720744094031385
            },
            "BD 0-20": {
                "value": 0.9579099485772622
            },
            "BD 10-20": {
                "value": 1.0303346946715117
            }
        }
    },
    "phy_attributes_texture": {
        "model_accuracy": {
            "Clay": {
                "MAE": 3.409383823913931
            },
            "Coarse": {
                "MAE": 3.6300433156846554
            },
            "Sand": {
                "MAE": 6.556105156001842
            },
            "Silt": {
                "MAE": 7.527928426229167
            }
        },
        "model_info": "Support Vector Regression (SVR) models",
        "result": {
            "Clay": {
                "value": 0.11518890679777094
            },
            "Coarse": {
                "value": 0.10196677734238974
            },
            "Sand": {
                "value": 0.1148972737558332
            },
            "Silt": {
                "value": 0.115200302211244
            }
        }
    }
}
```

