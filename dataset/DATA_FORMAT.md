# Format for the data

| Field                  | Type                         | Description                                                                                           |
| ---------------------- | ---------------------------- | ----------------------------------------------------------------------------------------------------- |
| `unique_id`            | `string`                     | MD5 hash of the question text (used as a unique identifier).                                          |
| `question`             | `string`                     | The full question text.                                                                               |
| `category`             | `string`                     | The subject or category of the question.                                                              |
| `choices`              | `array of strings` or `null` | List of answer choices for multiple-choice; `null` for open-ended.                                    |
| `choice_index_correct` | `integer` or `null`          | Index of the correct choice (starting from 0); `null` for open-ended.                                 |
| `explanation_correct`  | `string` or `null`           | Explanation or solution; `null` if not available.                                                     |
| `answer_correct`       | `string`                     | Final correct answer For multiple choice, this is the actual value of `choices[choice_index_correct]` |