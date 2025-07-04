# Format for the data

Intended to be stored in jsonl.

| Field                  | Type                         | Description                                                                                                            |
| ---------------------- | ---------------------------- | ---------------------------------------------------------------------------------------------------------------------- |
| `unique_id`            | `string`                     | MD5 hash of the *entire* record (all other columns), computed on the canonical JSON representation with sorted keys.   |
| `original_dataset`     | `string`                     | The name of the orginal dataset that the question came from.                                                           |
| `question`             | `string`                     | The full question text.                                                                                                |
| `choices`              | `array of strings` \| `null` | List of answer choices for multiple-choice questions; `null` for open-ended questions.                                 |
| `choice_index_correct` | `integer` \| `null`          | Index of the correct choice (0-based). Use `null` for open-ended questions.                                            |
| `explanation_correct`  | `string` \| `null`           | Explanation or worked solution. Use `null` if no explanation is available.                                             |
| `answer_correct`       | `string`                     | The final correct answer. For multiple-choice questions, this must equal the value of `choices[choice_index_correct]`. |

# Intended Structure

project-root/
├── converters/               # Dataset conversion scripts
│   ├── math_500.py           # The math 500 dataset
│   ├── mmlu_pro.py           # The mmlu pro dataset 
│   └── ...                   # Additional datasets
├── data/                     # All data
│   ├── raw/                  # Original datasets
│   ├── converted/            # Standardized JSONL
│   └── combined.jsonl        # Final deduplicated dataset
├── scripts/                  # Management scripts
│   ├── convert_all.py        # Run all converters
│   └── combine.py            # Merge datasets
├── utils.py                  # All utilities (single file)
├── requirements.txt          # Dependencies
└── README.md                 # Documentation