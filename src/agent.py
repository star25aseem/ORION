class OrionAgent:
    def __init__(self, model, uncertainty_threshold=0.02):
        self.model = model
        self.threshold = uncertainty_threshold

    def decide(self, prediction, uncertainty):
        if uncertainty < self.threshold:
            return {"decision": "ACCEPT", "reason": "Low uncertainty"}
        elif uncertainty < self.threshold * 2:
            return {"decision": "RE-EVALUATE", "reason": "Moderate uncertainty"}
        else:
            return {"decision": "REJECT", "reason": "High uncertainty"}


class OrionChatAgent:
    def __init__(self, threshold=0.35):
        self.threshold = threshold

    def decide(self, response, uncertainty):
        if uncertainty < self.threshold:
            return "ANSWER"
        elif uncertainty < self.threshold * 1.5:
            return "ASK_CLARIFICATION"
        else:
            return "REFUSE"


