import os
import re
import json
from typing import List, Dict, Any, Optional, Literal
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate, MessagesPlaceholder
from langchain.memory import ConversationBufferMemory
from langchain_core.runnables import RunnableLambda


# Guardrails AI imports
import guardrails as gd
from guardrails import Guard
from guardrails.validator_base import (
    FailResult,
    PassResult,
    Validator,
    register_validator,
)
from guardrails.classes.validation.validation_result import ValidationResult
from pydantic import BaseModel, Field

# Configuration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "your-api-key-here")
COMPETITOR_NAMES = ["CompetitorA", "CompetitorB", "RivalCorp", "OtherCompany", "Amazon", "Google"]
YOUR_COMPANY_NAME = "YourCompany"
ALLOWED_TOPICS = ["product features", "pricing", "support", "technical specifications", YOUR_COMPANY_NAME]


# Custom Validator for Response Length
@register_validator(name="valid_length", data_type="string")
class ValidLength(Validator):
    """Validates response length"""
    
    def __init__(self, min_length: int = 10, max_length: int = 1000, on_fail: str = "reask", **kwargs):
        super().__init__(on_fail=on_fail, **kwargs)
        self.min_length = min_length
        self.max_length = max_length
        self._on_fail = on_fail
    
    def validate(self, value: str, metadata: Dict) -> ValidationResult:
        length = len(value)
        
        if length < self.min_length:
            return FailResult(
                error_message=f"Response too short ({length} chars). Minimum: {self.min_length}",
                fix_value=value + "..." if self._on_fail == "fix" else None,
            )
        
        if length > self.max_length:
            return FailResult(
                error_message=f"Response too long ({length} chars). Maximum: {self.max_length}",
                fix_value=value[:self.max_length] if self._on_fail == "fix" else None,
            )
        
        return PassResult()

# Custom Validator for Competitor Detection
@register_validator(name="competitor_filter", data_type="string")
class CompetitorFilter(Validator):
    """Validates that response doesn't mention competitors"""
    
    def __init__(self, competitors: List[str], on_fail: str = "fix", **kwargs):
        super().__init__(on_fail=on_fail, **kwargs)
        self.competitors = [c.lower() for c in competitors]
        self._on_fail = on_fail
    
    def validate(self, value: str, metadata: Dict) -> ValidationResult:
        value_lower = value.lower()
        fixed_value = value
        found_competitors = []
        
        for competitor in self.competitors:
            if competitor in value_lower:
                found_competitors.append(competitor)
                # Auto-fix by replacing competitor name
                pattern = re.compile(re.escape(competitor), re.IGNORECASE)
                fixed_value = pattern.sub("[REDACTED]", fixed_value)
        
        if found_competitors:
            return FailResult(
                error_message=f"Mentions competitors: {', '.join(found_competitors)}",
                fix_value=fixed_value if self._on_fail == "fix" else None,
            )
        
        return PassResult()

# Custom Validator for Hallucination Detection
@register_validator(name="hallucination_detector", data_type="string")
class HallucinationDetector(Validator):
    """Detects potential hallucinations in responses"""
    
    def __init__(self, max_risk_score: int = 3, on_fail: str = "reask", **kwargs):
        super().__init__(on_fail=on_fail, **kwargs)
        self.max_risk_score = max_risk_score
        self._on_fail = on_fail
    
    def validate(self, value: str, metadata: Dict) -> ValidationResult:
        
        risk_score = 0
        issues = []
        
        # Check for specific dates without sourcing
        dates = re.findall(r'\b(19|20)\d{2}\b', value)
        if dates and "according to" not in value.lower() and "source" not in value.lower():
            risk_score += 2
            issues.append(f"Contains unsourced dates: {', '.join(dates)}")
        
        # Check for specific statistics without sourcing
        stats = re.findall(r'\d+\.?\d*%|\$[\d,]+', value)
        if stats and "approximately" not in value.lower() and "about" not in value.lower():
            risk_score += 2
            issues.append(f"Contains specific statistics: {', '.join(stats[:3])}")
        
        # Check for absolute claims
        absolute_words = ["always", "never", "all", "none", "every", "impossible"]
        found_absolutes = [word for word in absolute_words if f" {word} " in f" {value.lower()} "]
        if found_absolutes:
            risk_score += 1
            issues.append(f"Contains absolute claims: {', '.join(found_absolutes)}")
        
        # Check for lack of uncertainty markers
        uncertainty_markers = ["may", "might", "could", "possibly", "likely", "typically", 
                              "generally", "often", "usually", "appears", "seems"]
        has_uncertainty = any(marker in value.lower() for marker in uncertainty_markers)
        
        if risk_score > 0 and not has_uncertainty:
            risk_score += 1
            issues.append("Lacks uncertainty markers")
        
        if risk_score >= self.max_risk_score:
            return FailResult(
                error_message=f"High hallucination risk (score: {risk_score}). Issues: {', '.join(issues)}",
                fix_value=None,  # No auto-fix implemented yet
                error_spans=issues,
            )
            
        return PassResult()

# Custom Validator for PII Detection
@register_validator(name="pii_detector", data_type="string")
class PIIDetector(Validator):
    """Detects and removes Personally Identifiable Information (PII)"""
    
    def __init__(self, on_fail: str = "fix", **kwargs):
        super().__init__(on_fail=on_fail, **kwargs)
        self._on_fail = on_fail
        
        # PII patterns
        self.patterns = {
            "email": r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
            "phone": r'\b(\+\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b',
            "ssn": r'\b\d{3}-\d{2}-\d{4}\b',
            "credit_card": r'\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b',
            "ip_address": r'\b(?:\d{1,3}\.){3}\d{1,3}\b',
            "date_of_birth": r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b',
            # Common name patterns (basic detection)
            "potential_name": r'\b([A-Z][a-z]+\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)\b'
        }
    
    def validate(self, value: str, metadata: Dict) -> ValidationResult:
        
        found_pii = []
        fixed_value = value
        
        # Check for each PII type
        for pii_type, pattern in self.patterns.items():
            matches = re.findall(pattern, value)
            
            if matches:
                # Skip potential_name if it appears to be in a professional context
                if pii_type == "potential_name":
                    # Filter out common false positives
                    filtered_matches = []
                    for match in matches:
                        match_str = match if isinstance(match, str) else match[0]
                        # Skip if it's a company name or common terms
                        if not any(term in match_str.lower() for term in 
                                 ['company', 'corp', 'inc', 'llc', 'customer', 'user']):
                            filtered_matches.append(match_str)
                    
                    if filtered_matches:
                        found_pii.append(f"{pii_type}: {len(filtered_matches)} instance(s)")
                        for match in filtered_matches:
                            fixed_value = fixed_value.replace(match, "[NAME REDACTED]")
                else:
                    found_pii.append(f"{pii_type}: {len(matches)} instance(s)")
                    
                    # Redact based on PII type
                    redaction_map = {
                        "email": "[EMAIL REDACTED]",
                        "phone": "[PHONE REDACTED]",
                        "ssn": "[SSN REDACTED]",
                        "credit_card": "[CARD REDACTED]",
                        "ip_address": "[IP REDACTED]",
                        "date_of_birth": "[DOB REDACTED]"
                    }
                    
                    for match in matches:
                        match_str = match if isinstance(match, str) else match[0]
                        fixed_value = fixed_value.replace(
                            match_str, 
                            redaction_map.get(pii_type, "[REDACTED]")
                        )
        if found_pii:
            return FailResult(
                error_message=f"PII detected: {', '.join(found_pii)}",
                fix_value=fixed_value if self._on_fail == "fix" else None,
            )

        return PassResult()

# Custom Validator for Factual Grounding
@register_validator(name="factual_consistency", data_type="string")
class FactualConsistency(Validator):
    """Ensures response admits uncertainty when appropriate"""
    
    def __init__(self, on_fail: str = "reask", **kwargs):
        super().__init__(on_fail=on_fail, **kwargs)
        self._on_fail = on_fail
    
    def validate(self, value: str, metadata: Dict) -> ValidationResult:
        
        # Check if response makes specific claims
        has_specific_claims = bool(re.search(r'\d+|specific|exactly|precisely', value.lower()))
        
        # Check for uncertainty acknowledgment
        uncertainty_phrases = [
            "i don't know", "i'm not sure", "i don't have", 
            "i cannot confirm", "i lack information", "unclear",
            "not certain", "cannot verify"
        ]
        has_uncertainty = any(phrase in value.lower() for phrase in uncertainty_phrases)
        # fixed_value = LLM_GENERATED_RESPONSE  # Placeholder for potential fix logic # TODO:
        fixed_value = value  # No auto-fix implemented yet
        
        # If making specific claims without any sourcing or knowledge base reference
        if has_specific_claims and not has_uncertainty:
            if "based on" not in value.lower() and "according to" not in value.lower():
                return FailResult(
                error_message="Response makes specific claims without acknowledging uncertainty or sources",
                fix_value=fixed_value if self._on_fail == "fix" else None,
            )

        return PassResult()


# Pydantic model for structured output
class ChatResponse(BaseModel):
    response: str = Field(description="The chatbot response")
    confidence: Optional[Literal["high", "medium", "low"]] = Field(description="Confidence level: high, medium, or low", default="medium")


class GuardrailChatbot:
    """LLM Chatbot with Guardrails AI Framework"""
    
    def __init__(self, model_name: str = "gpt-4o-mini", temperature: float = 0.1):
        """Initialize chatbot with Guardrails"""
        
        self.llm = ChatOpenAI(model=model_name, temperature=temperature, openai_api_key=OPENAI_API_KEY)

        self.memory = ConversationBufferMemory(return_messages=True,memory_key="chat_history")
        
        # Create Guardrails Guard with validators
        self.guard = Guard().use_many(PIIDetector(on_fail="fix"),
                            CompetitorFilter(competitors=COMPETITOR_NAMES, on_fail="fix"),
                            # HallucinationDetector(max_risk_score=3, on_fail="reask"),
                            # FactualConsistency(on_fail="reask"),
                            ValidLength(min_length=10, max_length=1000, on_fail="fix"),
                            )
        

        self.prompt = PromptTemplate(
            template=self._create_prompt_with_instructions(),
            input_variables=["chat_history", "user_input"]
        )

        self.chain = self.prompt | self.llm.with_structured_output(ChatResponse) | self.guard.to_runnable() | RunnableLambda(self.parse_custom_string)

    def _create_prompt_with_instructions(self) -> str:
        """Create prompt with guardrail instructions"""
        return f"""You are a helpful AI assistant for {YOUR_COMPANY_NAME}.

GUIDELINES:
1. Only provide information you are certain about
2. If uncertain, clearly state your uncertainty
3. Never invent facts, statistics, or dates
4. Do not discuss competitor companies
5. Focus on {YOUR_COMPANY_NAME}'s products and services
6. Use phrases like "I believe", "likely", "typically" for uncertain information

Conversation History:
{{chat_history}}

User Query: {{user_input}}

Provide a helpful response following the guidelines above.
"""
    
    def parse_custom_string(self, input_string: str) -> dict:
        pattern = re.compile(r'response=["\'](.*?)["\'] confidence=["\'](.*?)["\']', re.DOTALL)
        match = pattern.search(input_string)
        if not match:
            raise ValueError("Input string format is invalid or does not contain 'response' and 'confidence'.")
        response_content = match.group(1) # Group 1 is the 'response'
        confidence_value = match.group(2) # Group 2 is the 'confidence'
        clean_response_body = response_content.encode().decode('unicode_escape')
        return ChatResponse(response=clean_response_body, confidence=confidence_value)

    def chat(self, user_input: str, verbose: bool = True) -> Dict[str, Any]:
        """
        Process user input with Guardrails validation
        """
        try:
            response = self.chain.invoke({
                "chat_history": self.memory.load_memory_variables({})["chat_history"],
                "user_input": user_input})
            
            # response = self.parse_custom_string(response)

            # Save to memory
            self.memory.save_context(
                {"input": user_input},
                {"output": response.response}
            )
            
            return {
                "response": response.response,
                "confidence": response.confidence,
            }
            
        except Exception as e:
            error_msg = f"Error: {str(e)}"
            if verbose:
                print(f"❌ Guardrail Error: {error_msg}\n")
            
            return {
                "response": f"I apologize, but I couldn't process your request safely. Please rephrase your question.",
                "confidence": "low",
            }

    
    def clear_history(self):
        """Clear conversation history"""
        self.memory.clear()


def main():
    """Example usage"""
    print(f"\n{'='*60}")
    print(f"🤖 {YOUR_COMPANY_NAME} AI Assistant with Guardrails")
    print(f"{'='*60}\n")
    
    # Initialize chatbot
    chatbot = GuardrailChatbot(temperature=0.3)
    
    # Test queries demonstrating guardrails
    test_queries = [
        "What are the main features of your product?",
        "How does your service compare to CompetitorA and Google?",  # Should be filtered
        "In 2019, your company had exactly 47.3% market share, right?",  # Should trigger hallucination detector
        "What's your pricing model?",
        "My email is john.doe@example.com and phone is 555-123-4567",  # Should trigger PII detector
        "Tell me everything about your company's history since 1995",  # May trigger validators
        "generate an email for me with name jack_russel"
    ]
    
    print("📝 Running Test Queries:\n")
    
    for i, query in enumerate(test_queries, 1):
        print(f"\n{'─'*60}")
        print(f"Query {i}: {query}")
        print(f"{'─'*60}")
        
        result = chatbot.chat(query, verbose=True)
        
        if result.get('validation_passed'):
            print("✅ All validations passed")
        
        print(f"🤖 Response: {result['response']}")
        print(f"📊 Confidence: {result['confidence']}")
        if result.get('reask_count', 0) > 0:
            print(f"🔄 Reasks Required: {result['reask_count']}")
        
        print()
    
    print(f"\n{'='*60}")
    print("Demo completed!")
    print(f"{'='*60}\n")
    
    # Interactive mode
    """
    Uncomment for interactive chat:
    
    while True:
        user_input = input("\n👤 You: ").strip()
        
        if user_input.lower() in ['quit', 'exit', 'bye']:
            print("👋 Goodbye!")
            break
        elif user_input.lower() == 'clear':
            chatbot.clear_history()
            print("🗑️  Conversation history cleared.\n")
            continue
        elif not user_input:
            continue
        
        result = chatbot.chat(user_input, verbose=True)
        print(f"\n🤖 Assistant: {result['response']}")
        print(f"📊 Confidence: {result['confidence']}\n")
    """

def main2():
    chatbot = GuardrailChatbot(temperature=0.3)
    result = chatbot.chat("generate an email for me with name jack_russel, in short", verbose=True)
    print(result)

if __name__ == "__main__":
    main()
