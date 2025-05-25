# ML Inversion Attack Project - Collaboration Guide

This guide outlines the recommended project structure and Git workflow for collaborative development of an ML inversion attack project with training, attack, and defense components.

## Project Structure

```
ml-inversion-project/
├── src/
│   ├── training/
│   │   ├── __init__.py
│   │   ├── trainer.py
│   │   ├── models.py
│   │   └── data_loader.py
│   ├── attack/
│   │   ├── __init__.py
│   │   ├── label_inference.py
│   │   └── attack_utils.py
│   ├── defense/
│   │   ├── __init__.py
│   │   ├── defense_mechanisms.py
│   │   └── defense_utils.py
│   └── utils/
│       ├── __init__.py
│       ├── common.py
│       └── visualization.py
├── scripts/
│   ├── train.py
│   ├── test_attack.py
│   ├── test_defense.py
│   └── run_experiments.py
├── configs/
│   ├── training_config.yaml
│   ├── attack_config.yaml
│   └── defense_config.yaml
├── tests/
├── requirements.txt
├── README.md
└── .gitignore
```

## Git Workflow Strategy

### 1. Branch Structure

```bash
main                    # Stable, tested code
├── feature/training    # Friend's training work
├── feature/attack      # Your attack implementation
└── feature/defense     # Your defense implementation
```

### 2. Initial Setup

```bash
# Create and switch to feature branches
git checkout -b feature/training
git checkout -b feature/attack
git checkout -b feature/defense
```

### 3. Define Clear Interfaces Early

Create a shared interface contract that both developers agree on. Start with skeleton files in `main`:

#### src/training/trainer.py
```python
class ModelTrainer:
    def __init__(self, config):
        pass
    
    def train_model(self):
        """Returns trained model"""
        raise NotImplementedError
    
    def save_model(self, path):
        raise NotImplementedError
    
    def load_model(self, path):
        raise NotImplementedError
```

#### src/attack/label_inference.py
```python
class LabelInferenceAttack:
    def __init__(self, model, config):
        pass
    
    def perform_attack(self, data):
        """Returns attack results"""
        raise NotImplementedError
```

### 4. Collaboration Best Practices

#### For Minimal Conflicts:
- **Separate config files**: Each module has its own config
- **Mock/stub dependencies**: Use placeholder functions initially
- **Frequent small commits**: Commit often with clear messages
- **Regular syncing**: Pull from main frequently

#### Daily Workflow:
```bash
# Daily routine for each developer
git checkout feature/your-branch
git pull origin main        # Get latest changes
git merge main             # Merge main into your branch
# Work on your code
git add .
git commit -m "Clear commit message"
git push origin feature/your-branch
```

### 5. Integration Strategy

#### Phase 1: Parallel Development with Mocks
```python
# In your attack code, mock the training module
def get_trained_model():
    # Return dummy model for testing
    return MockModel()
```

#### Phase 2: Integration Testing
- Create integration branch: `git checkout -b integration`
- Merge both feature branches
- Test end-to-end functionality

#### Phase 3: Production Merge
- Only merge to main after thorough testing

### 6. Additional Recommendations

#### Use GitHub Features:
- **Pull Requests**: Review each other's code
- **Issues**: Track tasks and bugs
- **Project Board**: Organize work

#### Code Standards:
```python
# Consistent imports and structure
from src.training.trainer import ModelTrainer
from src.utils.common import load_config

# Clear function signatures
def label_inference_attack(model, target_data, attack_params):
    """
    Perform label inference attack
    
    Args:
        model: Trained model
        target_data: Data to attack
        attack_params: Attack configuration
    
    Returns:
        AttackResults object
    """
```

#### Testing Strategy:
- Unit tests for individual modules
- Integration tests for module interactions
- Use pytest for consistent testing

## Work Division

### Friend's Responsibilities (Training):
- Implement CIFAR-10 data loading and preprocessing
- Create model architectures
- Implement training and fine-tuning procedures
- Model saving/loading functionality

### Your Responsibilities (Attack & Defense):
- Implement label inference attack mechanisms
- Create defense mechanisms against attacks
- Integration testing between attack and defense
- Performance evaluation metrics

## Key Success Factors

1. **Define interfaces early** and stick to them
2. **Communicate frequently** about changes
3. **Test individually** before integration
4. **Use descriptive commit messages**
5. **Review each other's code** via pull requests

This structure allows independent work while maintaining a clear integration path and minimizing merge conflicts.