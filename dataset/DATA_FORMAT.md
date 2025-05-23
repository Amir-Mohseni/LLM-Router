# Format for the data

| Field                  | Type                         | Description                                                                                                                        |
| ---------------------- | ---------------------------- | ---------------------------------------------------------------------------------------------------------------------------------- |
| `unique_id`            | `string`                     | **MD5 hash of the *entire* record (all other columns), computed on the canonical JSON representation with sorted keys.**           |
| `question`             | `string`                     | The full question text.                                                                                                            |
| `category`             | `string`                     | The subject or category of the question.                                                                                           |
| `choices`              | `array of strings` \| `null` | List of answer choices for multiple-choice questions; `null` for open-ended questions.                                             |
| `choice_index_correct` | `integer` \| `null`          | Index of the correct choice (0-based). Use `null` for open-ended questions.                                                        |
| `explanation_correct`  | `string` \| `null`           | Explanation or worked solution. Use `null` if no explanation is available.                                                         |
| `answer_correct`       | `string`                     | The final correct answer. For multiple-choice questions, this must equal `choices[choice_index_correct]`.                           |
