from models.schemas import LLMTrainingSchema
from src.services.interfaces import LLMReasoningEngine


class StubLLMReasoningEngine(LLMReasoningEngine):
    async def analyze_description(
        self,
        text: str,
        description_id: str,
        classification_cat: str,
    ) -> tuple[LLMTrainingSchema, dict]:
        lowered = text.lower()
        thoughts = []
        actions = []

        thoughts.append("Check image classification result.")
        if classification_cat == "Flagged (Unsafe)":
            output = "Policy Violation"
            thoughts.append("Unsafe image detected.")
            actions.append("Block post.")
            return (
                LLMTrainingSchema(
                    llm_train_id=f"llm-{description_id}",
                    output=output,
                    is_correct=None,
                    accuracy=None,
                    description_key=description_id,
                ),
                {
                    "reasoning_type": "ReAct",
                    "thoughts": thoughts,
                    "actions": actions,
                },
            )

        thoughts.append("Check description text safety.")
        if any(word in lowered for word in ["hate", "kill", "violent"]):
            output = "Policy Violation"
            thoughts.append("Unsafe text detected.")
            actions.append("Block post.")
        else:
            output = "Approved"
            thoughts.append("Content appears safe.")
            actions.append("Approve post.")

        return (
            LLMTrainingSchema(
                llm_train_id=f"llm-{description_id}",
                output=output,
                is_correct=None,
                accuracy=None,
                description_key=description_id,
            ),
            {
                "reasoning_type": "ReAct",
                "thoughts": thoughts,
                "actions": actions,
            },
        )
