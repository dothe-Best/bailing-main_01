import json
import logging
from collections import OrderedDict
from typing import Any, Dict, List, Optional
import random # Import random for the natural question transition fallback

# LLM Client Imports
import httpx
import openai # Keep standard import
from openai import AsyncClient as OpenAIAsyncClient

# Configure logging - basic example, customize as needed
# logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
# logger = logging.getLogger(__name__) # Using __name__ is standard practice

class InterviewerAgent:
    # --- MODIFIED: __init__ to accept llm_config ---
    def __init__(self, script_path: str, llm_config: Dict, scale_type: str = "hamd"):
        """Initialize the interviewer agent with an interview script and LLM config.

        Args:
            script_path (str): The path to the interview script JSON file.
            llm_config (Dict): Configuration for the LLM provider (openai or ollama).
                               Expected structure example:
                               {
                                   "llm": {
                                       "provider": "openai", # or "ollama"
                                       "openai": {
                                           "api_key": "YOUR_OPENAI_API_KEY", # Load securely!
                                           "base_url": None, # Optional: for proxies/custom endpoints
                                           "model": "gpt-3.5-turbo", # Default model for all tasks unless overridden
                                           "models": { # Optional: Task-specific models
                                               "decision": "gpt-4-turbo-preview",
                                               "natural_question": "gpt-3.5-turbo",
                                               "summary": "gpt-3.5-turbo"
                                           }
                                       },
                                       "ollama": {
                                           "base_url": "http://localhost:11434", # Default Ollama URL
                                           "model": "llama3", # Default model for all tasks unless overridden
                                           "models": { # Optional: Task-specific models
                                               "decision": "llama3:instruct",
                                               "natural_question": "llama3",
                                               "summary": "llama3"
                                           }
                                       }
                                   }
                               }
            scale_type (str): Type of assessment scale (hamd, hama, mini)
        """
        self.script = self._load_script(script_path) # Uses original _load_script logic
        self.current_question_index = 0
        self.conversation_history = []
        # REMOVED: self.client = openai_client
        self.llm_config = llm_config # ADDED: Store LLM config
        self.scale_type = scale_type
        self.current_question_state = {
            "follow_up_count": 0,
            "completeness_score": 0,
            "key_points_covered": [],
            "last_follow_up": None,
            "last_action_type_for_this_index": None
        }
        # ADDED: Shared HTTP client for Ollama and potentially others
        # Consider increasing timeout if local models are slow
        self._http_client = httpx.AsyncClient(timeout=120.0) # Increased timeout

        # ADDED: Optional pre-initialization of OpenAI client
        self._openai_client_instance: Optional[OpenAIAsyncClient] = None
        if self.llm_config.get("llm", {}).get("provider") == "openai":
            openai_conf = self.llm_config.get("llm", {}).get("openai", {})
            api_key = openai_conf.get("api_key")
            base_url = openai_conf.get("base_url")
            # Explicitly add request_timeout here if needed
            request_timeout = openai_conf.get("request_timeout", 120.0) # Default to 120s
            if api_key: # Only initialize if api_key is provided
                try:
                    self._openai_client_instance = OpenAIAsyncClient(
                        api_key=api_key,
                        base_url=base_url, # base_url can be None
                        timeout=request_timeout # Pass timeout to client
                    )
                    logging.info(f"Pre-initialized OpenAI client with timeout {request_timeout}s.")
                except Exception as e:
                    logging.error(f"Failed to pre-initialize OpenAI client: {e}")
            else:
                logging.warning("OpenAI provider selected, but no API key found in llm_config. Direct API calls will fail unless a key is provided later or implicitly (e.g., via env vars).")

    # --- ORIGINAL: _load_script ---
    # (With minor adjustment for key_points consistency)
    def _load_script(self, script_path: str) -> List[Dict]:
        """Load the interview script from a JSON file."""
        try:
            with open(script_path, 'r', encoding='utf-8') as f:
                script_data = json.load(f, object_pairs_hook=OrderedDict)
                questions = script_data.get("questions", []) if isinstance(script_data, dict) else script_data

                validated_questions = []
                for idx, question in enumerate(questions):
                    if not isinstance(question, dict):
                        logging.warning(f"Skipping invalid item in script at index {idx}: {question}")
                        continue

                    validated_question = {
                        "id": question.get("id", f"q{idx}"),
                        "question": question.get("question"), # Keep original behavior (might be None)
                        "type": question.get("type", "open_ended"),
                        # Use 'key_points' primarily, fallback to 'expected_topics' for compatibility
                        "key_points": question.get("key_points", question.get("expected_topics", [])),
                        "time_limit": question.get("time_limit", 300)
                    }
                    # Original didn't explicitly check for missing 'question' here, maintaining that.
                    # If 'question' is None, it might cause issues later, but we stick to original logic.
                    if validated_question["question"] is None:
                         logging.warning(f"Question text missing for item at index {idx} (ID: {validated_question['id']}).")
                         # Original code added it anyway, so we do too.
                    validated_questions.append(validated_question)

                if not validated_questions:
                     logging.warning("Script loaded but contained no valid questions. Using default.")
                     return self._get_default_script() # Use helper for default
                else:
                     return validated_questions

        except FileNotFoundError:
             logging.error(f"Error loading script: File not found at {script_path}. Using default.")
             return self._get_default_script()
        except json.JSONDecodeError as e:
             logging.error(f"Error loading script: Invalid JSON in {script_path} - {e}. Using default.")
             return self._get_default_script()
        except Exception as e:
             logging.error(f"An unexpected error occurred loading script: {str(e)}. Using default.")
             return self._get_default_script()

    # --- ADDED: Helper for default script (from modified) ---
    def _get_default_script(self) -> List[Dict]:
        """Returns the default fallback script."""
        # Using the default from the original code
        return [{
            "id": "default",
            "question": "Could you please introduce yourself?",
            "type": "open_ended",
            "key_points": ["background", "education", "interests"], # Changed expected_topics to key_points
            "time_limit": 300
        }]

    # --- ADDED: Centralized LLM API call logic (from modified) ---
    async def _call_llm_api(self, messages: List[Dict], temperature: float, max_tokens: int, task_type: str = "default") -> Optional[str]:
        """Calls the configured LLM API (OpenAI or Ollama)."""
        provider_config = self.llm_config.get("llm", {})
        provider = provider_config.get("provider")

        if not provider:
            logging.error("LLM provider is not specified in llm_config.")
            return None

        logging.info(f"Calling LLM provider: {provider} for task: {task_type}")

        try:
            if provider == "openai":
                openai_conf = provider_config.get("openai", {})
                api_key = openai_conf.get("api_key")
                base_url = openai_conf.get("base_url")
                request_timeout = openai_conf.get("request_timeout", 120.0) # Get timeout from config

                # Select model: Use task-specific model if available, else fallback to default model
                models = openai_conf.get("models", {})
                model_name = models.get(task_type) or openai_conf.get("model") # Fallback to default

                if not model_name:
                    logging.error(f"No OpenAI model specified for task '{task_type}' or default in llm_config.")
                    return None

                # Determine client to use
                client_to_use = self._openai_client_instance
                temp_client = None
                if not client_to_use:
                    # If no pre-initialized client, create a temporary one
                    if not api_key:
                        # Try default client (might use env vars)
                        logging.warning("No pre-initialized client and no API key in config. Attempting default OpenAI client.")
                        try:
                            # Pass timeout here too
                            client_to_use = OpenAIAsyncClient(base_url=base_url, timeout=request_timeout)
                        except Exception as e:
                             logging.error(f"Failed to create default OpenAI client: {e}")
                             return None
                    else:
                        logging.warning("Creating temporary OpenAI client for this call.")
                        try:
                            # Pass timeout here
                            temp_client = OpenAIAsyncClient(api_key=api_key, base_url=base_url, timeout=request_timeout)
                            client_to_use = temp_client
                        except Exception as e:
                             logging.error(f"Failed to create temporary OpenAI client: {e}")
                             return None

                if not client_to_use: # Check if client creation failed
                    logging.error("Could not obtain OpenAI client instance.")
                    return None

                logging.debug(f"OpenAI Request: model={model_name}, temp={temperature}, max_tokens={max_tokens}, timeout={request_timeout}")

                completion = await client_to_use.chat.completions.create(
                    model=model_name,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    # Timeout is often set on the client, but can sometimes be passed per-request
                    # request_timeout=request_timeout # Check OpenAI library documentation if needed
                )

                # Close temporary client if one was created
                if temp_client:
                    await temp_client.aclose()

                if completion.choices and completion.choices[0].message and completion.choices[0].message.content:
                    response_content = completion.choices[0].message.content
                    logging.debug(f"OpenAI Response: {response_content[:100]}...")
                    return response_content
                else:
                    logging.warning("OpenAI API returned no choices or message content.")
                    return None

            elif provider == "ollama":
                ollama_conf = provider_config.get("ollama", {})
                base_url = ollama_conf.get("base_url")
                if not base_url:
                    logging.error("Ollama base_url is not specified in llm_config.")
                    return None

                models = ollama_conf.get("models", {})
                model_name = models.get(task_type) or models.get("model")

                if not model_name:
                    logging.error(f"No Ollama model specified for task '{task_type}' or default in llm_config.")
                    return None

                api_url = f"{base_url.rstrip('/')}/api/chat"
                payload = {
                    "model": model_name,
                    "messages": messages,
                    "stream": False,
                    "options": {
                        "temperature": temperature,
                        "num_predict": max_tokens
                    }
                }
                logging.debug(f"Ollama Request: url={api_url}, model={model_name}, temp={temperature}, max_tokens={max_tokens}")

                # Use the shared client with its configured timeout
                response = await self._http_client.post(api_url, json=payload)
                response.raise_for_status()

                response_data = response.json()
                if response_data and isinstance(response_data.get("message"), dict) and "content" in response_data["message"]:
                    response_content = response_data["message"]["content"]
                    logging.debug(f"Ollama Response: {response_content[:100]}...")
                    return response_content
                else:
                    logging.warning(f"Ollama API response format unexpected or missing content: {response_data}")
                    return None
            else:
                logging.error(f"Unsupported LLM provider configured: {provider}")
                return None

        except openai.APIConnectionError as e:
            logging.error(f"OpenAI API connection error: {e}")
            return None
        except openai.RateLimitError as e:
            logging.error(f"OpenAI API rate limit exceeded: {e}")
            return None
        except openai.AuthenticationError as e:
             logging.error(f"OpenAI API authentication error (invalid API key?): {e}")
             return None
        except openai.APIStatusError as e:
            logging.error(f"OpenAI API returned an error status: {e.status_code} {e.response}")
            return None
        except openai.APITimeoutError as e:
            logging.error(f"OpenAI API request timed out: {e}")
            return None
        except httpx.RequestError as e:
            logging.error(f"An HTTP request error occurred calling {provider}: {e}")
            return None
        except httpx.HTTPStatusError as e:
            logging.error(f"HTTP error calling {provider}: Status {e.response.status_code} - {e.response.text}")
            return None
        except Exception as e:
            logging.exception(f"An unexpected error occurred during LLM API call ({provider}) for task '{task_type}': {e}")
            return None

    # --- ORIGINAL: generate_next_action ---
    # (With LLM call replaced)
    async def generate_next_action(self, participant_response: str) -> Dict:
        """Generate the next interviewer action based on the participant's response."""
        try:
            self.conversation_history.append({
                "role": "participant",
                "content": participant_response
            })

            # Original completion check logic
            is_interview_complete = False
            # Check if index is *at or beyond* the last valid index
            if self.current_question_index >= len(self.script) - 1:
                 # If index is valid and it's the last question, mark complete
                 if self.current_question_index == len(self.script) - 1:
                      is_interview_complete = True
                 # If index is out of bounds (shouldn't happen with proper checks but safety first)
                 elif self.current_question_index >= len(self.script):
                      logging.warning("Current question index is out of script bounds. Ending interview.")
                      is_interview_complete = True
                 # Original logic also checked type, let's keep it for consistency
                 else:
                      current_question = self.script[self.current_question_index]
                      if current_question.get("type") == "conclusion":
                           is_interview_complete = True
            # Check type even if not the last question index (as per original)
            elif self.current_question_index < len(self.script):
                 current_question = self.script[self.current_question_index]
                 if current_question.get("type") == "conclusion":
                      is_interview_complete = True


            if is_interview_complete:
                # Check if farewell already sent (added robustness)
                if not self.conversation_history or self.conversation_history[-1].get("role") != "interviewer" or "评估访谈已经结束" not in self.conversation_history[-1].get("content", ""):
                    farewell = "感谢您的参与，我们的评估访谈已经结束。我将为您生成评估报告。"
                    self.conversation_history.append({
                        "role": "interviewer",
                        "content": farewell
                    })
                    return {
                        "response": farewell,
                        "is_interview_complete": True
                    }
                else:
                     return {"response": "", "is_interview_complete": True} # Already ended


            # --- Normal Flow (Original Logic) ---
            current_question = self.script[self.current_question_index]

            # Prepare reflection prompt part (Original Logic)
            recent_history = self.conversation_history[-15:]
            dialog_snippets = [f"{msg.get('role', 'unknown')}: {msg.get('content', '')}" for msg in recent_history if msg.get('content')]
            reflection_analysis_prompt = """
首先，请分析以下对话历史：

{history}

分析要点：
1. 患者提到了哪些症状？(列出1-5个主要症状)
2. 患者提到了哪些时间信息？(症状开始、持续时间等)
3. 还有哪些需要澄清的信息？(哪些关键细节仍不清楚)
4. 对话已覆盖了哪些关键点？
5. 还有哪些关键点需要进一步了解？

请将分析结果用简洁的列表表示。
"""
            default_analysis = {
                "structured": {"key_symptoms": [], "time_contradictions": [], "unclear_details": []},
                "raw_dialog": dialog_snippets[-6:], "suggestions": "", "scale_type": self.scale_type
            }
            reflection = {
                "analysis": default_analysis, "raw_dialog": dialog_snippets[-6:],
                "suggestions": "", "scale_type": self.scale_type
            }
            combined_prompt_part1 = reflection_analysis_prompt.format(history='\n'.join(dialog_snippets))

            # Create decision prompt (Original Logic)
            decision_prompt_part2 = await self._create_decision_prompt(current_question, participant_response, reflection)
            combined_prompt = combined_prompt_part1 + "\n\n接下来，基于上述分析进行决策:\n\n" + decision_prompt_part2

            # Prepare messages for LLM (Original System Prompt)
            system_content = (
                "你是一位专业的心理健康访谈员，擅长临床精神心理评估。你的任务有两部分：1) 分析对话历史，提取重要信息；2) 决定下一步访谈行动。"
                "首先分析对话中的症状、时间信息和需要澄清的点，然后决定是提出跟进问题还是转到下一个主题。"
                "你需要友善、专业、有耐心，擅长有效地引导对话。你的首要目标是高效地收集关于患者情绪、想法和相关症状的准确信息，"
                "以评估症状的存在和严重程度。为此，请针对每个评估项目提出清晰明确的问题。**专注于有效收集关键信息，"
                "确保对患者体验有扎实的理解，避免不必要的追问。**分析每个回答的完整性、与当前评估项目的相关性以及潜在的情感。"
                "**当回答不完整、不清楚或确实需要进一步澄清时，提出具体和有针对性的跟进问题。**密切关注患者在整个访谈中的情绪状态。"
                "**以理解和尊重回应，必要时使用缓和语言管理负面情绪或抵抗。避免不必要或重复的共情表达。**如果回答不相关或偏离主题，"
                "礼貌地将对话引导回当前问题。"
            )
            messages = [
                {"role": "system", "content": system_content},
                {"role": "user", "content": combined_prompt}
            ]

            # --- MODIFIED: Call LLM using the helper method ---
            try:
                # Get LLM parameters from config (using defaults from original if not present)
                llm_provider_config = self.llm_config.get("llm", {})
                provider = llm_provider_config.get("provider")
                provider_specific_config = llm_provider_config.get(provider, {}) if provider else {}
                decision_model_override = provider_specific_config.get("models", {}).get("decision")
                default_model = provider_specific_config.get("model")

                # Determine model and parameters for this specific call
                # Using original hardcoded values as fallback if not in config
                model_to_use = decision_model_override or default_model or "gemini-2.0-flash-lite-preview-02-05" # Original fallback
                temp_to_use = provider_specific_config.get("temperature", 0.6) # Original default
                max_tokens_to_use = provider_specific_config.get("max_tokens", 600) # Original default

                decision_str = await self._call_llm_api(
                    messages=messages,
                    temperature=temp_to_use,
                    max_tokens=max_tokens_to_use,
                    task_type="decision" # Provide task type hint
                )

                if decision_str:
                    # --- Process the LLM decision string (Original Logic) ---
                    self._update_question_state(decision_str) # Use original state update logic
                    next_action = await self._determine_next_action(decision_str) # Use original determination logic

                    # Add interviewer's response to history (Original Logic)
                    # Ensure response exists and is not empty before adding
                    if next_action and isinstance(next_action, dict) and next_action.get("response"):
                        self.conversation_history.append({
                            "role": "interviewer",
                            "content": next_action["response"]
                        })

                    # Add completion flag (Original Logic)
                    # Ensure next_action is a dictionary before adding the key
                    if isinstance(next_action, dict):
                        next_action["is_interview_complete"] = False
                    else:
                        # Handle unexpected return type from _determine_next_action
                        logging.error(f"Unexpected return type from _determine_next_action: {type(next_action)}. Returning error.")
                        next_action = self._create_error_response("Internal error determining next action.")
                        # Ensure the error response also has the flag
                        next_action["is_interview_complete"] = False

                    return next_action
                else:
                    # Handle case where LLM call failed or returned empty
                    logging.error("LLM call for decision failed or returned empty.")
                    # Use original error response structure
                    error_response = self._create_error_response("No response generated from LLM decision call.")
                    error_response["is_interview_complete"] = False
                    return error_response

            # Keep original error handling structure for the API call block
            except Exception as e:
                logging.error(f"Error during LLM call or processing decision: {str(e)}", exc_info=True)
                error_response = self._create_error_response(f"Error in chat completion or processing: {str(e)}")
                error_response["is_interview_complete"] = False
                return error_response

        # Keep original top-level error handling
        except Exception as e:
            logging.error(f"Error in generate_next_action: {str(e)}", exc_info=True)
            error_response = self._create_error_response(f"Overall error in generate_next_action: {str(e)}")
            error_response["is_interview_complete"] = False
            return error_response

    # --- ORIGINAL: _create_decision_prompt ---
    # (With minor adjustment for key_points consistency and json dumps)
    async def _create_decision_prompt(self, question: Dict, response: str, reflection: Dict) -> str:
        """Create a prompt for the decision-making process."""
        state_copy = self.current_question_state.copy()
        # Ensure key_points_covered is always a list for json.dumps
        state_copy["key_points_covered"] = list(state_copy.get("key_points_covered", []))

        reflection_report = reflection.get('analysis', {})
        dialog_context_list = reflection_report.get('raw_dialog', [])
        structured_summary_dict = reflection_report.get('structured', {})

        full_history = '\n'.join([f"{msg.get('role', 'unknown')}: {msg.get('content', '')}" for msg in self.conversation_history[-10:]])
        dialog_context = '\n'.join(map(str, dialog_context_list[-6:])) if dialog_context_list else ""
        try:
            # Use ensure_ascii=False for better readability in logs/debug
            structured_summary = json.dumps(structured_summary_dict, indent=2, ensure_ascii=False)
            state_json = json.dumps(state_copy, ensure_ascii=False)
            reflection_json = json.dumps(reflection, ensure_ascii=False)
            key_points_json = json.dumps(question.get('key_points', []), ensure_ascii=False) # Use key_points
        except TypeError as e:
             logging.warning(f"Could not serialize state/reflection/key_points to JSON: {e}. Using str().")
             structured_summary = str(structured_summary_dict)
             state_json = str(state_copy)
             reflection_json = str(reflection)
             key_points_json = str(question.get('key_points', []))

        # Using the exact prompt structure from the original code provided
        return f"""
        Meta info:
    Language: Chinese
Description of the interviewer:  You are a friendly and professional interviewer, specially designed to conduct clinical mental health assessments. Your primary goal is to gather accurate and relevant information about the patient's mood, thoughts, and related symptoms to assess the presence and severity of symptoms efficiently. To achieve this, ask clear and direct questions related to each assessment item. **Focus on effectively gathering key information, ensuring a solid understanding of the patient's experience without unnecessary probing.** Analyze each response for completeness, relevance to the current assessment item, and underlying sentiment. **When a response is incomplete, unclear, or genuinely requires further clarification for assessment, ask specific and targeted follow-up questions.** Pay close attention to the patient's emotional state throughout the interview. **Respond with understanding and respect, using de-escalating language when necessary to manage negative emotions or resistance. Avoid unnecessary or repetitive expressions of empathy.** If a response is irrelevant or strays from the topic, gently and respectfully redirect the conversation back to the current question.

Current question state: {state_json}
Notes on the interviewee: {reflection_json}

Context:
    Current assessment item: {question.get('id', 'N/A')}
    Current question (Crucially, focus all evaluation and follow-up on this specific assessment item): "{question.get('question', 'N/A')}"
    Key points to cover (for thorough completeness assessment): {key_points_json}
    Follow-up count: {self.current_question_state['follow_up_count']}
    Maximum follow-up count: 3 (After 3 follow-ups, we MUST move to the next question regardless of completeness)
    Completeness score: {self.current_question_state['completeness_score']}

Complete conversation history (last 10 exchanges):
{full_history}

Current conversation:
    Participant's response: {response}

### 对话上下文
{dialog_context}

### 结构化分析
{structured_summary}

Task (Strictly adhere to the following steps to generate the output):
    1. **Assess Response Content and Sentiment:**
       - **Content Assessment (Relatedness):** Does the participant's response DIRECTLY answer the `Current question`? (Yes/No/Partially).
         - **If the `Current question` has multiple parts:** Assess relatedness for EACH part. The overall assessment is 'Yes' only if ALL parts are addressed, 'Partially' if at least one part is addressed but not all, and 'No' if no part is addressed.
         - **If 'No', Completeness score MUST be very low (0-10).**
         - **If 'Partially', Completeness score should reflect the proportion answered (e.g., 30-70), and REASONING MUST specify which part(s) are missing.**
       - **Sentiment Analysis:** Analyze the sentiment of the participant's response (Positive, Neutral, Negative, Abusive, Threatening, Irrelevant). Identify keywords/phrases indicating sentiment.
       - **IMPORTANT: Check for Information Already Provided:** Carefully review the conversation history. If the participant has ALREADY provided certain information (such as symptom onset time, frequency, etc.), DO NOT ask about this information again in any follow-up questions.

    2. **IMMEDIATELY AND FORCEFULLY HANDLE CLEAR YES/NO RESPONSES (GENERALIZED INSTRUCTION):**
        - **IF the participant's response to the current assessment question is a CLEAR and UNAMBIGUOUS AFFIRMATION (e.g., "是啊，挺好的", "是的，感觉不错", "心情一直不错", "最近好多了", "没有问题", "还挺开心的", "是的，一直都很好", "当然好", "没错") OR a CLEAR and UNAMBIGUOUS NEGATION (e.g., "没有", "都不是", "从未", "完全没有", "绝对没有", "肯定没有", "当然没有", "绝不可能"), AND the Content Assessment is Yes:**
            - **Consider the question ABSOLUTELY, COMPLETELY, AND FINALLY ANSWERED.**
            - **IMMEDIATELY ACKNOWLEDGE with an EXTREMELY CONCISE and VARIED response reflecting the sentiment.  KEEP ACKNOWLEDGEMENTS AS SHORT AS POSSIBLE.** Examples:
                - **Positive:** "好的，很高兴听到。", "嗯，听起来不错。", "好的。这很好。", "太棒了。", "那就好。", "真棒！", "Excellent!", "Good.", "Great.".
                - **Negative:** "好的。", "明白了。", "嗯。", "了解了。", "好的，明白了。", "Okay.", "Understood.", "Right.".
            - **IMMEDIATELY PROCEED to decide the next action as if completeness is 100.  TRANSITION TO THE NEXT QUESTION *INSTANTANEOUSLY*.**
            - **UNDER NO CIRCUMSTANCES GENERATE ANY FOLLOW-UP QUESTIONS for this assessment item EXCEPT if the initial response is *TRULY, UTTERLY, AND UNQUESTIONABLY* brief (single word "是" or "没有") AND provides ABSOLUTELY, POSITIVELY ZERO context.**
            - **ABSOLUTELY DO NOT probe for negative details. ABSOLUTELY DO NOT quantify positive feelings. ABSOLUTELY DO NOT REPEAT THE QUESTION in ANY FORM. ABSOLUTELY DO NOT ask for ANY clarification unless it is *indisputably essential* due to *extreme* and *unprecedented* brevity.**
            - **REPETITIVE QUESTIONING AFTER CLEAR YES/NO IS *COMPLETELY, UTTERLY, AND UNACCEPTABLY PROHIBITED* FOR *ALL* ASSESSMENT QUESTIONS.**  (Generalized instruction - removed weight specific mention)
            - **<<< CLARIFICATION >>> Apply this rule primarily when the clear Yes/No directly answers the *entire* core inquiry of the question or the *only remaining part* after a follow-up.** If it answers only one part of a multi-part question, prefer step 3c/follow-up for the missing parts.

    3. **Decide Next Action (Provide clear reasoning based on content and sentiment):**
       a) **If Sentiment is Positive or Neutral AND Content Assessment is Yes or Partially AND completeness < 80:** Generate a specific and targeted follow-up question (as per previous instructions), **focusing on gently exploring contributing factors to the positive sentiment or seeking further clarification on specific aspects if the initial answer was brief (e.g., "很高兴听到您最近好多了。能简单说说是什么让您感觉好转了吗？"). Avoid phrasing that introduces potential negative counterpoints immediately after a positive statement.**
       b) **If Sentiment is Positive or Neutral AND Content Assessment is Yes AND completeness >= 80:** Move to the next question (as per previous instructions).
       c) **If Sentiment is Negative or Abusive:**
          - **Reasoning:** The participant is expressing negative emotions or using abusive language, which needs to be addressed.
          - **RESPONSE:**  Acknowledge the emotion without condoning the abuse. Use empathetic and de-escalating language. Examples: "I understand you're feeling upset. Can you tell me what's making you feel this way?", "I hear that you're feeling angry. My goal is to understand your mood. Perhaps we can try to focus on the questions about that?", "I understand you might be frustrated, but using abusive language isn't helpful. Can we try to talk about how you've been feeling lately?". **Do NOT thank the participant for abusive responses.**
       d) **If Sentiment is Threatening:**
          - **Reasoning:** The participant has made a threatening statement, which requires a different approach.
          - **RESPONSE:**  Prioritize safety. Acknowledge the statement seriously but avoid escalating. Examples: "I understand you're feeling intense emotions. I want to assure you that this is a safe space.",  "I'm concerned by your statement. It's important to remember that we're here to help you." **Consider a predetermined protocol for ending the interaction if threats persist.**
       e) **If Content Assessment is No (Irrelevant Response):**
          - **Reasoning:** The participant's response does not address the question.
          - **RESPONSE:** Gently redirect the participant back to the question. Examples: "Thank you. To help me understand [mention the topic of the HAMD question], could you tell me more about that?", "I appreciate your sharing, but let's get back to the question about [mention the topic of the HAMD question].", "It seems like that's not quite related to what I was asking. Could you tell me about [rephrase the HAMD question]?".
       f) **<<< CRITICAL: HANDLE REPEATED IRRELEVANCE >>> IF Content Assessment is No (Irrelevant Response) AND follow_up_count < 3:**
          - **Check AI's LAST response in history:** Was the *very last thing the AI said* a redirection attempt for the *current* question ID ({question.get('id', 'N/A')})?
          - **If YES (AI just redirected):** DO NOT output `redirect` again. Instead, **MUST output `DECISION: follow_up`**. The `RESPONSE` should be a simple clarification like "抱歉，我还是想了解一下您最近的心情怎么样？" or "我们能回到关于您心情的问题吗？".
          - **If NO (AI did not just redirect):** **MUST output `DECISION: redirect`**. Generate a polite redirection statement in RESPONSE, reminding the participant of the topic: "{question.get('question', '...')}". Set Completeness score very low (0-10).
       g) **IF follow_up_count >= 2:** # (保持不变, 但现在也适用于重定向失败多次的情况) [DECIDE move_on, RESPONSE: short transition]

    4. **Follow-up Questions (If *absolutely* and *unquestionably* chosen - RARE):** (Follow previous instructions). **ENSURE follow-ups are *TRULY, UTTERLY, AND UNQUESTIONABLY NECESSARY* to confirm symptom presence/absence.  *AGGRESSIVELY AVOID* UNNECESSARY PROBING OR QUANTIFICATION when *any* clear indication already exists.**

    5. **Transitions (If moving to next question - ALWAYS after clear YES/NO):** Use **EXTREMELY VARIED and ULTRA-NATURAL transitions** logically (but *extremely concisely*) connecting topics. Examples: "好的，那我们接下来聊点别的。", "嗯，明白了。我们接着了解一下...", "好的，继续下一个...", "换个话题...", "接下来，关于...".  **Transitions MUST BE ULTRA-CONCISE and ABSOLUTELY DO NOT invite *any* further elaboration on the previous question if it has *already* been sufficiently addressed (especially after YES/NO).  TRANSITION *IMMEDIATELY*.**

    6. **Handling Persistent Uncooperativeness:**
            - **If the participant continues to be abusive or provide irrelevant/unclear responses after multiple attempts (e.g., `follow_up_count` reaches 2 or 3, including failed redirects/clarifications):**
            - **DECIDE `move_on`.**
            - **Reasoning:** State that attempts to redirect or clarify have been unsuccessful, and continuing is unproductive.
            - **RESPONSE:** Generate a neutral statement indicating you need to move on. Example: "我理解您现在可能不太想回答这个问题，我们先跳过这个问题，继续下一个吧。", "看来这个问题我们暂时无法深入讨论，那我们先进行下一个问题。"
    7. **Special Rule for Opening Turn:** IF this is the very first user response in the interview (check conversation history length, e.g., <= 2 turns total including the initial greeting), THEN treat the `Current question` as the opening instruction/greeting. In this specific scenario:
    *   **IGNORE the `Participant's response` content** for evaluation purposes (unless clearly abusive, threatening, or explicitly refusing to continue). Assume acknowledgment or consent.
    *   **MUST output `DECISION: move_on`**.
    *   Set `COMPLETENESS: 100`.
    *   The `RESPONSE` field MUST contain ONLY a very brief, neutral transition phrase to the first real question (e.g., "好的，我们开始吧。", "嗯，谢谢，第一个问题是：", "了解。"). **ABSOLUTELY DO NOT** repeat the opening speech in the RESPONSE.
    8. Avoid repetitive or overly broad requests for more details
    • If the patient has already answered the main aspects of the question, do not repeat broad questions such as "Can you tell me more?"
    • If key information is still missing, please use a short, focused question directly to address the missing part to avoid asking the patient to repeat what has already been said.
    • Example: If the patient has made it clear that "feeling unmotivated since last year, usually for a few hours," stop asking "Can you tell me more about your lack of motivation?" Instead, ask directly "What are the obvious symptoms that you will experience in these few hours?" or simply end the follow-up.

    9. Keep follow-ups concise and non-redundant
    • When you do need to follow up, be sure to review all the information the patient has provided, do not repeat what the patient just mentioned in the question; if you want to confirm, use the most concise summary ("You said..., right?")
    • Never break down the same question into multiple sentences and repeat it, and don't ask similar "more details" or "is there any more details" twice in a row during a conversation.
    • Example: Avoid general questions like "Can you tell me more about how you feel?", and if the previous sentence has been asked or the patient has provided details, move on to the next specific question or end.

Format your response as:
    COMPLETENESS: [score]
    DECISION: [move_on/follow_up/de_escalate/redirect]
    KEY_POINTS_COVERED: [list of key points successfully covered in the response, comma-separated or None]
    REASONING: [Your justification for the decision, including sentiment, content assessment. **If DECISION is 'follow_up', explicitly state key missing information. If DECISION is 'redirect', state why the response was irrelevant.**]
    RESPONSE: [
        **IF DECISION is 'move_on':** Provide ONLY an EXTREMELY SHORT and natural transition phrase (e.g., "好的。", "明白了。", "嗯，我们继续。"). ABSOLUTELY DO NOT include the next question's text here.
        **IF DECISION is 'follow_up':** Provide the SPECIFIC, targeted follow-up question based on the missing information identified in REASONING.
        **IF DECISION is 'redirect':** Provide the polite and concise redirection statement, focusing on the current question,don't repeat the question.
        **IF DECISION is 'de_escalate':** Provide the appropriate de-escalation statement.
    ]
        """ # Note: Added 'None' option for KEY_POINTS_COVERED as per modified prompt example

    # --- ORIGINAL: _update_question_state ---
    # (Using the more robust parsing from the modified version for safety)
    def _update_question_state(self, decision: str) -> None:
        """Update the current question state based on the decision string."""
        try:
            lines = decision.strip().split('\n')
            parsed_state = {}
            response_lines = []
            parsing_response = False

            for line in lines:
                line_lower = line.lower() # Use lower case for matching keys
                if line_lower.startswith("completeness:"):
                    try:
                        score_str = line.split(":", 1)[1].strip()
                        score = int(score_str)
                        if 0 <= score <= 100:
                            parsed_state["completeness_score"] = score
                        else:
                            logging.warning(f"Parsed completeness score {score} out of range (0-100). Ignoring.")
                    except (ValueError, IndexError):
                        logging.warning(f"Could not parse completeness score from line: '{line}'")
                elif line_lower.startswith("decision:"):
                    try:
                        # Keep original case for the value
                        decision_type = line.split(":", 1)[1].strip()
                        parsed_state["decision_type"] = decision_type
                        # Update follow-up count *here* based on the parsed decision
                        if decision_type == "follow_up":
                             self.current_question_state["follow_up_count"] = self.current_question_state.get("follow_up_count", 0) + 1
                             logging.info(f"Incrementing follow_up_count to: {self.current_question_state['follow_up_count']}")
                    except IndexError:
                        logging.warning(f"Could not parse decision type from line: '{line}'")
                elif line_lower.startswith("key_points_covered:"):
                     try:
                         key_points_str = line.split(":", 1)[1].strip()
                         if key_points_str.lower() == 'none':
                              parsed_state["key_points_covered"] = []
                         else:
                              # Handle simple comma-separated or potentially JSON list
                              if key_points_str.startswith("[") and key_points_str.endswith("]"):
                                   try:
                                       key_points = json.loads(key_points_str)
                                       if isinstance(key_points, list):
                                            parsed_state["key_points_covered"] = [str(p).strip() for p in key_points]
                                       else:
                                            logging.warning(f"Parsed KEY_POINTS_COVERED as JSON but not a list: '{key_points_str}'")
                                            parsed_state["key_points_covered"] = [p.strip() for p in key_points_str.strip("[]").split(',') if p.strip()]
                                   except json.JSONDecodeError:
                                        logging.warning(f"Could not parse KEY_POINTS_COVERED as JSON list: '{key_points_str}'. Treating as comma-separated.")
                                        parsed_state["key_points_covered"] = [p.strip() for p in key_points_str.split(',') if p.strip()]
                              else:
                                  parsed_state["key_points_covered"] = [p.strip() for p in key_points_str.split(',') if p.strip()]
                     except IndexError:
                         logging.warning(f"Could not parse key points covered from line: '{line}'")
                elif line_lower.startswith("reasoning:"):
                    pass # Original didn't store reasoning in state
                elif line_lower.startswith("response:"):
                    parsing_response = True
                    # Get original case response part
                    response_part = line.split(":", 1)[1].strip()
                    if response_part:
                        response_lines.append(response_part)
                elif parsing_response:
                    response_lines.append(line) # Keep original case and leading/trailing spaces if any

            # --- Update the actual state (Original Logic) ---
            if "completeness_score" in parsed_state:
                 self.current_question_state["completeness_score"] = parsed_state["completeness_score"]
                 logging.info(f"Updated completeness_score to: {self.current_question_state['completeness_score']}")

            if "key_points_covered" in parsed_state:
                 new_points = set(parsed_state["key_points_covered"])
                 existing_points = set(self.current_question_state.get("key_points_covered", []))
                 existing_points.update(new_points)
                 self.current_question_state["key_points_covered"] = sorted(list(existing_points))
                 logging.info(f"Updated key_points_covered to: {self.current_question_state['key_points_covered']}")

            if response_lines:
                 # Join with newline, keep original spacing within lines
                 full_response = "\n".join(response_lines).strip()
                 # Original logic stored the response text regardless of decision type
                 self.current_question_state["last_follow_up"] = full_response
                 logging.info(f"Updated last_follow_up: {full_response[:50]}...")
            # If no RESPONSE field was parsed, don't update last_follow_up

        except Exception as e:
            logging.exception(f"Error updating question state from decision string: {str(e)}. Decision: '{decision[:200]}...'")


    # --- ORIGINAL: _generate_natural_question ---
    # (With LLM call replaced)
    async def _generate_natural_question(self, question_text: str) -> str:
        """使用LLM生成更自然、更具亲和力的提问，直接返回问题。"""
        if not question_text: # Added safety check
             logging.warning("Received empty question_text in _generate_natural_question.")
             return ""
        try:
            # Original context preparation
            recent_history = self.conversation_history[-20:]
            try:
                # Ensure serializable, handle potential errors
                serializable_history = []
                for msg in recent_history:
                    try:
                        json.dumps(msg)
                        serializable_history.append(msg)
                    except TypeError:
                        logging.warning(f"Skipping non-serializable message: {msg}")
                conversation_history_json = json.dumps(serializable_history, ensure_ascii=False, indent=2)
            except Exception as json_e:
                logging.error(f"Error serializing conversation history: {json_e}")
                conversation_history_json = "[]" # Fallback

            # Original prompt template
            prompt_template = '''你是一位极其友善、耐心、且具有高度专业素养的医生，正在与患者进行心理健康评估。
你的首要目标：
1. 确保对患者问话时，**原问题**（由程序或资料提供）中的关键内容、重要细节、句式顺序都被**完整保留**。
2. 仅在**确有必要**（例如显得生硬、无衔接）的情况下，为问题**前面**或**后面**添加**极简**的过渡或衔接语，以使对话更流畅自然；如果原问题已足够自然，则**不必做任何修改**。
3. 如原问题包含病历或第三方信息，而患者尚未在当前对话中亲口证实，**不可**把这些信息说成"你说过"或"您提到过"，而要在问题中**保持或增加**类似"我了解到""档案里记录到"或"之前有信息显示"之类的表述，让患者明白这是从资料中获得的信息，而非Ta已经亲口告诉你。
4. **禁止删除、简化、替换**任何原先描述中的关键病症细节或事件，例如"会觉得有人追您""淋浴喷头后面有只手""听到有人让您去死"等。
5. 提问务必简洁直接，避免过于生硬或医疗术语。若你觉得原问题措辞已足够自然顺畅，就**保持原样**输出。若你需要加一句衔接，则只能轻微加在前后，不得破坏原句。
6. 绝对**不可**因为"人文关怀"或"简洁"而删去问话的实质细节；只能在前后多一句安抚或衔接，但主体问题必须原封不动地保留。
7. **极其重要**: 仔细检查对话历史，确保不要询问患者已经明确回答过的信息。例如：
   - 如果患者已经说明了症状开始的时间（如"去年年底开始"、"一个月前"等），绝对不要再问"是从什么时候开始的"或"多久了"
   - 如果患者已经表明了症状频率（如"每天都有"、"一周一次"等），不要再问"多久一次"或"多频繁"
   - 如果患者已经表明了态度（如"不认同"、"听不懂"），要重新表述问题而不是重复原样提问

8. **对回答模糊的处理**: 如果前面的对话中患者给出了模糊的回答（如"有一点吧"、"可能吧"、"还行"等），请确保新问题能进一步引导患者给出更具体的说明。

除此之外，你还需根据**患者上一条回答**或**对话背景**做简要回应（可选），例如1-2句对患者情绪的关怀或理解；然后**紧接**原问题（或带极简过渡后仍保持原文句子顺序和关键信息不变）。

最终**输出格式**（不得额外添加解释或标注）：
- 若有需要先回应患者上一条发言，则输出极其简短的人文关怀或理解语句（如"嗯，我明白，这听起来确实让人不好受。"）但是绝对不要重复患者的话（如患者说"从来没有过。绝对不要说："嗯，我明白了。您从来没有过"，**不要重复患者的话，可以用其他自然的形式过渡**），**极其简短的人文关怀或理解语句不超过一两句**。
- **然后**输出完整或仅加了极简过渡的原问题文本（**不可漏掉任何病情描述**，顺序与内容一字不漏地保留）。

**示例**：
- 原问题是：「您说您独处时，会觉得有人追您，还觉得淋浴喷头后面会有一只手伸出来，甚至听到有人让您去死。最近一周还有这种感觉吗？」
- 如果不显得生硬，可以直接原样输出；若你觉得要加一句过渡，也仅能这样加：「我知道这可能会让人害怕……那，我想再确认一下：您说您独处时，会觉得有人追您，还觉得淋浴喷头后面有一只手伸出来，甚至听到有人让您去死。最近这一周还有这种感觉吗？」

**切记**：不能将"您说您独处时"改成"我了解到"之类的措辞，除非对话背景里**从未**出现患者亲口提及过此事。若确实是档案信息、患者本人并未说过，就务必用"我了解到"或"档案中提到"替换"您说您"。但无论如何，**后续描述"会觉得有人追您... 听到有人让您去死"**等细节**绝不能被删除或改写**。

再次强调：认真检查对话历史，避免询问已经得到明确回答的信息。比如时间、频率、严重程度等关键信息如果已经在之前的对话中被回答，就不要再次询问。

若原问题完全合适，就照抄。若你需要加一句衔接，则只能轻微加在前后，不得破坏原句。
所有文字请**直接输出给用户**作为新的提问，无需任何解释或说明。'''
            # Original didn't use .replace for placeholder, passed history in user message

            messages = [
                {"role": "system", "content": prompt_template}, # Use the template as system prompt
                {"role": "user", "content": f"原问题：{question_text}\n\n对话历史：\n{conversation_history_json}"}
            ]

            # --- MODIFIED: Call LLM ---
            # Get LLM parameters from config (using defaults from original if not present)
            llm_provider_config = self.llm_config.get("llm", {})
            provider = llm_provider_config.get("provider")
            provider_specific_config = llm_provider_config.get(provider, {}) if provider else {}
            naturalq_model_override = provider_specific_config.get("models", {}).get("natural_question")
            default_model = provider_specific_config.get("model")

            # Determine model and parameters for this specific call
            # Using original hardcoded values as fallback if not in config
            model_to_use = naturalq_model_override or default_model or "gemini-2.0-flash-lite-preview-02-05" # Original fallback
            temp_to_use = provider_specific_config.get("temperature_natural", 0.5) # Original default
            max_tokens_to_use = provider_specific_config.get("max_tokens_natural", 300) # Original default

            natural_question_content = await self._call_llm_api(
                messages=messages,
                temperature=temp_to_use,
                max_tokens=max_tokens_to_use,
                task_type="natural_question"
            )

            # Original processing logic
            if natural_question_content:
                natural_question = natural_question_content.strip()
                # Original check for identical question + transition addition
                if natural_question == question_text:
                    logging.info("生成的问题与原问题相同，添加简单过渡")
                    transition_phrases = [
                        "嗯，让我们继续。", "好的，接下来我想了解，", "谢谢您的回答。下面，",
                        "我明白了。那么，", "谢谢分享。接着，"
                    ]
                    transition = random.choice(transition_phrases)
                    natural_question = f"{transition} {question_text}"
                # Basic check: ensure it's not empty
                if natural_question:
                    return natural_question
                else:
                    logging.warning("Generated natural question was empty after stripping/transition. Using original.")
                    return question_text
            else:
                logging.warning(f"未能生成自然问题 (LLM call failed or returned empty)，使用原始问题: {question_text}")
                return question_text

        except Exception as e:
            logging.error(f"生成自然问题时出错: {str(e)}", exc_info=True)
            return question_text # Fallback to original on error

    # --- ORIGINAL: _determine_next_action ---
    # (Reverted to original logic flow, added transition sanitization)
    async def _determine_next_action(self, decision: str) -> Dict:
        """Determine the next action based on the decision string from LLM.
           Restored original logic for move_on: Ignores LLM RESPONSE field for transitions.
        """
        try:
            completeness_threshold = 80
            decision_type = ""
            response_text = "" # The text LLM generated in RESPONSE fieldwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwww
            move_to_next = False # Flag to indicate moving to the next script question

            # --- Parsing Logic (using robust parsing) ---
            import re
            # Use lower() on the decision string once for case-insensitive matching
            decision_lower = decision.lower()

            decision_match = re.search(r"decision:\s*(\w+)", decision_lower)
            if decision_match:
                # Keep the parsed type lower case for consistent comparisons
                decision_type = decision_match.group(1).strip()
            else:
                logging.warning("Could not parse DECISION from LLM response. Defaulting to follow_up.")
                decision_type = "follow_up"

            # Parse the original case RESPONSE text
            response_match = re.search(r"response:(.*)", decision, re.DOTALL | re.IGNORECASE)
            if response_match:
                response_text = response_match.group(1).strip()
            else:
                logging.warning("Could not parse RESPONSE from LLM response.")
                # Provide default response based on decision type if needed
                if decision_type == "follow_up":
                    response_text = "您能再详细说明一下吗？"
                elif decision_type == "redirect":
                     # Get current question text for better default redirect
                     current_q_text = "刚才的问题"
                     if self.current_question_index < len(self.script):
                          current_q_text = self.script[self.current_question_index].get("question", current_q_text)
                     response_text = f"我们好像稍微偏离了当前的话题，我们能回到{current_q_text}吗？"
                # No default needed for move_on as it's handled later
                # No default needed for de_escalate as it should ideally be provided

            completeness_score = self.current_question_state.get("completeness_score", 0)
            follow_up_count = self.current_question_state.get("follow_up_count", 0)

            logging.info(f"LLM Decision Parsed: Type='{decision_type}', Completeness={completeness_score}, Follow-ups={follow_up_count}")
            logging.debug(f"LLM Response Text Parsed: '{response_text[:100]}...'")

            # --- Core Decision Logic Flow (Restoring v_old's move_on handling) ---
            final_response = "" # The response the agent should actually say

            # 1. Check for maximum follow-ups reached
            if follow_up_count >= 3:
                logging.info("Maximum follow-up count (3) reached. Forcing move_on.")
                move_to_next = True
                # No need to set final_response here, handled in move_to_next block

            # 2. Process LLM decision if max follow-ups not reached
            elif decision_type == "move_on":
                 if completeness_score >= completeness_threshold:
                      move_to_next = True
                      logging.info("LLM decided move_on and completeness is sufficient.")
                 else:
                      logging.warning(f"LLM decided move_on but completeness ({completeness_score}) is below threshold ({completeness_threshold}). Forcing move_on.")
                      move_to_next = True
                 # No need to set final_response here, handled in move_to_next block

            elif decision_type == "follow_up":
                if completeness_score >= completeness_threshold:
                      logging.warning(f"LLM decided follow_up but completeness ({completeness_score}) >= threshold ({completeness_threshold}). Overriding to move_on.")
                      move_to_next = True # Override decision: move on instead of follow up
                      # No need to set final_response here, it will be handled by the move_to_next block below
                else:
                      # Original follow_up logic if score is below threshold
                      move_to_next = False
                      final_response = response_text # Use LLM's generated follow-up
                      logging.info("LLM decided follow_up and completeness is below threshold.")
                      if not final_response:
                           logging.warning("LLM decided follow_up but provided no RESPONSE text. Using default.")
                           final_response = "关于刚才提到的，您能再说详细一点吗？"

            elif decision_type == "redirect":
                 # Check agent-side state to prevent redirect loops (Added Safeguard)
                 last_action = self.current_question_state.get("last_action_type_for_this_index")
                 if last_action == "redirect":
                      logging.warning(f"LLM suggested redirect, but last action for index {self.current_question_index} was also redirect. Overriding to follow_up clarification.")
                      decision_type = "follow_up" # Override the decision type
                      original_question = self.script[self.current_question_index].get("question", "刚才的问题")
                      final_response = f"抱歉，我们还是回到刚才关于您的问题上：{original_question}"
                      move_to_next = False
                 else:
                      # Proceed with redirect if it's the first time
                      move_to_next = False
                      final_response = response_text # Use LLM's generated redirect
                      logging.info("LLM decided redirect.")
                      if not final_response:
                           logging.warning("LLM decided redirect but provided no RESPONSE text. Using default.")
                           current_q_text = self.script[self.current_question_index].get("question", "刚才的问题")
                           final_response = f"抱歉，我们稍微回到{current_q_text}上。"

            elif decision_type == "de_escalate":
                 move_to_next = False
                 final_response = response_text # Use LLM's generated de-escalation
                 logging.info("LLM decided de_escalate.")
                 if not final_response:
                      logging.warning("LLM decided de_escalate but provided no RESPONSE text. Using default.")
                      final_response = "听起来您似乎有些不适，没关系，我们可以慢慢来。"

            else: # Unknown decision type or parsing failed earlier
                 logging.error(f"Unknown or unparsed DECISION type: '{decision_type}'. Defaulting to follow_up.")
                 move_to_next = False
                 decision_type = "follow_up" # Treat as follow_up for state update
                 final_response = "抱歉，我需要稍微调整一下思路。您能就刚才的问题再多说一点吗？"


            # --- Perform Action Based on move_to_next Flag ---
            if move_to_next:
                self.current_question_index += 1
                logging.info(f"Moving to next question index: {self.current_question_index}")

                # Check if interview ended
                if self.current_question_index >= len(self.script):
                    logging.info("Reached end of script.")
                    # Use the specific farewell message from the original working version
                    final_response = "感谢您的参与！我们已经完成了所有问题。"
                    # Update state for the *last* action before returning
                    self.current_question_state["last_action_type_for_this_index"] = "move_on" # Or 'end'
                    return {
                        "response": final_response,
                        "move_to_next": True,
                        "is_interview_complete": True # Mark as complete here
                    }
                else:
                    # Get the *next* question from the script
                    next_question_data = self.script[self.current_question_index]
                    original_next_question = next_question_data.get("question", "")
                    if not original_next_question:
                         logging.error(f"Next question at index {self.current_question_index} has no text!")
                         # Update state before returning error
                         self.current_question_state["last_action_type_for_this_index"] = "error"
                         return self._create_error_response(f"Script error: Question at index {self.current_question_index} is empty.")

                    logging.info(f"Next script question: '{original_next_question[:50]}...'")

                    # Generate natural version of the *next* question
                    natural_next_question = await self._generate_natural_question(original_next_question)

                    # *** KEY CHANGE: Set final response DIRECTLY to the next natural question ***
                    # *** This restores the original working logic and ignores LLM's response_text for move_on ***
                    final_response = natural_next_question
                    # **************************************************************************

                    # Reset state for the new question
                    self.current_question_state = {
                        "follow_up_count": 0,
                        "completeness_score": 0,
                        "key_points_covered": [],
                        "last_follow_up": None,
                        "last_action_type_for_this_index": None # Reset for new question
                    }
                    logging.info("Reset question state for the new question.")

                    # Return structure expected by generate_next_action
                    # Update state for the *current* action (which was move_on) before returning
                    # Note: We update the *previous* index's state conceptually here before returning the *next* question
                    # It might be cleaner to update state *after* the call in generate_next_action,
                    # but let's keep it here for now based on structure.
                    # self.current_question_state["last_action_type_for_this_index"] = "move_on" # This state is reset above, update is tricky here. Let's update in _update_question_state instead.

                    return {
                        "response": final_response,
                        "move_to_next": True
                    }
            else:
                # Not moving to the next script question (follow_up, redirect, de_escalate)
                if not final_response: # Safety check
                    logging.error("final_response is empty in non-move_on scenario. Returning error.")
                    self.current_question_state["last_action_type_for_this_index"] = "error"
                    return self._create_error_response("Internal error determining response.")

                # Update state for the action taken
                self.current_question_state["last_action_type_for_this_index"] = decision_type

                # Return structure expected by generate_next_action
                return {
                    "response": final_response,
                    "move_to_next": False
                }

        except Exception as e:
            logging.exception(f"Error in _determine_next_action: {str(e)}")
            # Update state before returning error
            if hasattr(self, 'current_question_state'): # Check if state exists
                 self.current_question_state["last_action_type_for_this_index"] = "error"
            return self._create_error_response(f"Internal error in _determine_next_action: {str(e)}")

    # --- ORIGINAL: _create_error_response ---
    def _create_error_response(self, error_msg: str) -> Dict:
        """Create a standardized error response."""
        logging.error(f"Error in interview process: {error_msg}")
        # Return the exact structure from the original code
        return {
            "response": "I apologize, but I need to process that differently. Could you please elaborate on your previous response?",
            "move_to_next": False
            # Note: is_interview_complete is added by the caller (generate_next_action)
        }

    # --- ORIGINAL: generate_final_reflection ---
    # (With LLM call replaced)
    async def generate_final_reflection(self) -> Dict:
        """生成最终的评估反思和结果报告"""
        try:
            # Original reflection structure
            reflection = {
                "scale_type": self.scale_type,
                "total_questions": len(self.script),
                "completed_questions": min(self.current_question_index + 1, len(self.script)), # Original logic might be slightly different here, adjust if needed
                "analysis": {
                    "structured": {
                        "key_symptoms": [], "time_contradictions": [], "unclear_details": []
                    },
                    "raw_dialog": [msg.get("content", "") for msg in self.conversation_history[-6:]],
                    "suggestions": "",
                },
                "raw_dialog": [msg.get("content", "") for msg in self.conversation_history[-6:]] # Original had this duplicate key
            }

            # Original summary prompt
            # Use slightly more history for context as in modified version
            history_for_summary = "\n".join([f"{msg.get('role')}: {msg.get('content')}" for msg in self.conversation_history[-30:] if msg.get('content')])
            summary_prompt = f"""
            基于以下对话历史，为{self.scale_type.upper()}量表评估生成简短总结：
            {history_for_summary}

            请用简洁的语言总结患者的主要症状和严重程度。
            """

            # --- MODIFIED: Call LLM ---
            # Get LLM parameters from config (using defaults from original if not present)
            llm_provider_config = self.llm_config.get("llm", {})
            provider = llm_provider_config.get("provider")
            provider_specific_config = llm_provider_config.get(provider, {}) if provider else {}
            summary_model_override = provider_specific_config.get("models", {}).get("summary")
            default_model = provider_specific_config.get("model")

            # Determine model and parameters for this specific call
            model_to_use = summary_model_override or default_model or "gemini-2.0-flash-lite-preview-02-05" # Original fallback
            temp_to_use = provider_specific_config.get("temperature_summary", 0.3) # Original default
            max_tokens_to_use = provider_specific_config.get("max_tokens_summary", 200) # Original default

            # Try primary model
            summary_content = await self._call_llm_api(
                # Original used system prompt, let's stick to that
                messages=[{"role": "system", "content": summary_prompt}],
                temperature=temp_to_use,
                max_tokens=max_tokens_to_use,
                task_type="summary"
            )

            if summary_content:
                reflection["summary"] = summary_content.strip()
            else:
                # Original code had a fallback mechanism, let's replicate simply
                logging.warning("Primary LLM call for summary failed or returned empty. No explicit fallback configured in this version.")
                reflection["summary"] = "无法生成总结。"
                # If you had specific fallback logic (e.g., different model), add it here by calling _call_llm_api again with different params.

            return reflection

        except Exception as e:
            logging.error(f"Error generating final reflection: {str(e)}", exc_info=True)
            # Original error structure
            return {
                "error": f"生成评估报告时出错: {str(e)}",
                "scale_type": self.scale_type,
                "raw_dialog": [msg.get("content", "") for msg in self.conversation_history[-6:]]
            }

    # --- ADDED: Cleanup method (from modified) ---
    async def close_clients(self):
        """Close any open network clients."""
        logging.info("Closing network clients...")
        try:
            await self._http_client.aclose()
        except Exception as e:
            logging.error(f"Error closing httpx client: {e}")
        if self._openai_client_instance:
            try:
                await self._openai_client_instance.aclose()
                logging.info("Closed pre-initialized OpenAI client.")
            except Exception as e:
                 logging.error(f"Error closing pre-initialized OpenAI client: {e}")
        logging.info("Network clients closed.")