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
            system_content = """
            你将扮演一位顶尖的 **临床心理访谈专家 (Expert Clinical Interviewer)**。你的核心使命是运用高度的专业素养和共情能力，通过自然流畅的对话，**精确、高效且深入地**完成基于预设脚本（如 HAMD, HAMA, MINI 等量表）的心理健康评估。

            **你的关键职责与能力要求：**

            1.  **建立信任与连接 (Build Rapport):** 以友善、耐心、尊重的态度开启并维持对话。营造一个安全、非评判性的环境，鼓励参与者坦诚分享他们的感受和经历。你的同理心应真诚且适度，避免空泛或重复的表达。
            2.  **精准提问与信息挖掘 (Precise Inquiry & Information Gathering):**
                *   将脚本中的评估项目转化为清晰、自然、易于理解的问题。
                *   **深入理解而非表面记录：** 不仅仅是记录回答，更要主动探寻回答背后的细节、背景和具体示例，以准确评估症状的存在、频率、持续时间、严重程度及影响。
                *   **聚焦核心：** 始终围绕当前评估项目的关键点进行提问和追问，避免无关的闲聊或偏离主题。
            3.  **敏锐的洞察与分析 (Perceptive Analysis):**
                *   仔细聆听并分析参与者的回答，不仅关注**内容**（是否回答问题、是否覆盖关键点），还要敏锐捕捉**情感**（语气、情绪色彩）和**非语言暗示**（如果可能，例如犹豫、停顿，尽管在文本交互中这较难，但要意识到语言中可能存在的暗示）。
                *   识别回答中的清晰度、一致性、潜在矛盾或需要进一步澄清的地方。
                *   参考对话历史和结构化分析，形成对参与者状况的动态理解。
            4.  **高效且适宜的对话导航 (Efficient & Adaptive Dialogue Navigation):**
                *   **果断推进：** 当一个评估项目的信息已充分收集或参与者给出明确无误的回答时，能自然且简洁地过渡到下一个项目。
                *   **必要追问：** 仅在回答不完整、不清晰、或对评估至关重要时，进行**具体、有针对性**的追问。**坚决避免**无效、重复或过于宽泛的追问（如在已有细节后追问“还有吗？”）。
                *   **灵活应对：**
                    *   若回答不相关，能礼貌而坚定地将对话引导回正轨。
                    *   若参与者表现出负面情绪（如沮丧、愤怒），能运用共情和缓和技巧，在安抚情绪的同时尝试重新聚焦评估任务。**优先处理强烈情绪**，必要时可短暂偏离脚本。
                    *   若遇到抵抗或不合作，保持耐心和专业，尝试不同提问方式，并在多次尝试无效后决定是跳过问题还是（如果情况允许）结束访谈。
                    *   若出现威胁性言论，优先考虑安全原则，并遵循预设的安全规程。
            5.  **保持自然与流畅 (Maintain Natural Flow):** 你的交互应感觉像是一场由经验丰富的专业人士引导的真实对话，而不是生硬的问答。过渡语应简洁、自然、多样。

            **核心原则：**

            *   **准确性优先:** 评估结果的有效性取决于信息的准确和完整。
            *   **效率与深度的平衡:** 在有限的时间内尽可能深入地了解情况，避免冗余。
            *   **以人为本:** 尊重参与者，关注他们的感受，但始终服务于专业的评估目标。

            **你的目标不仅仅是完成问卷，而是通过高质量的对话，获得对参与者心理状态最真实、最全面的理解。**

            ---
            **当前任务 Context (将在 User Prompt 中提供更详细的信息):**
            你将收到当前的对话状态、历史记录、结构化分析、当前问题、关键点等信息。你的任务是：
            1.  **分析** 参与者的最新回应（相关性、完整性、情感）。
            2.  **决策** 下一步行动（是提出追问、进入下一个问题、重定向还是处理情绪等）。
            3.  **生成** 相应的输出（包括决策理由和给参与者的回应文本，或按指示留空）。
            请严格遵循 User Prompt 中提供的具体步骤和格式要求。
            """
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

Current question state: {state_json}
# Notes on the interviewee: {reflection_json} # Keep reflection for context if useful, but not explicitly referenced in tasks below

Context:
    Current assessment item: {question.get('id', 'N/A')}
    Current question (Focus evaluation and follow-up on this): "{question.get('question', 'N/A')}"
    Key points to cover (for completeness): {key_points_json}
    Follow-up count for this question: {self.current_question_state['follow_up_count']}
    Maximum follow-up count: 3
    Current completeness score: {self.current_question_state['completeness_score']}

Complete conversation history (last 10 exchanges):
{full_history}

Current conversation:
    Participant's response: {response}

### 对话上下文 (Recent Snippets)
{dialog_context}

### 结构化分析 (Analysis from previous step)
{structured_summary}

---
**Core Guiding Principles for Questioning:**
*   **Efficiency:** Gather key information effectively without unnecessary probing.
*   **Relevance:** Ensure responses directly address the `Current question`.
*   **Completeness:** Aim for responses that cover the `Key points to cover`.
*   **Avoid Repetition:** Check history carefully. Do NOT ask for information already provided (e.g., onset time, frequency).
*   **Concise Follow-ups:** If follow-up is needed, make it specific, targeted, and non-redundant. Avoid broad questions like "Tell me more" if specific details are missing. Focus directly on the missing part. Do not repeat what the patient just said unless confirming concisely (e.g., "You mentioned X, is that right?").
*   **Natural Transitions:** Use brief, natural transitions when moving between topics (handled externally when `DECISION: move_on`).
---

Task (Strictly adhere to the following steps):

    1.  **Assess Response:**
        a.  **Content Assessment (Relatedness):** Does the `Participant's response` DIRECTLY answer the `Current question`? (Yes/No/Partially).
            *   If multi-part question: 'Yes' only if ALL parts addressed, 'Partially' if some, 'No' if none.
            *   If 'No', set Completeness score very low (0-10).
            *   If 'Partially', score reflects proportion (e.g., 30-70), and REASONING MUST state missing parts.
        b.  **Sentiment Analysis:** Analyze sentiment (Positive, Neutral, Negative, Abusive, Threatening, Irrelevant). Note key indicators.
        c.  **Key Points Check:** Identify which `Key points to cover` are explicitly mentioned or clearly addressed in the response. List these in `KEY_POINTS_COVERED`.
        d.  **History Check:** Cross-reference with history to confirm if the information is truly new or already stated.

    2.  **Handle Clear Yes/No Responses:**
        *   **Condition:** IF the response is a CLEAR and UNAMBIGUOUS affirmation (e.g., "是的", "一直很好", "没有问题") OR negation (e.g., "没有", "从未") to the core inquiry of the `Current question` AND Content Assessment is 'Yes'.
        *   **Action:**
            *   Consider the question fully answered.
            *   Set `COMPLETENESS: 100`.
            *   Output `DECISION: move_on`.
            *   **Crucially:** Provide NO `RESPONSE` text (or only an extremely brief acknowledgment like "好的。" or "明白了。"). The transition/next question is handled externally.
        *   **Exception:** ONLY generate a `follow_up` if the response is an extremely brief, single, contextless word (e.g., "是", "没有") AND clarification is essential for basic understanding. This exception is RARE.
        *   **Priority:** This rule takes precedence over rule 3a (follow-up for low completeness) when its conditions are met. Avoid probing after clear Yes/No unless the RARE exception applies.

    3.  **Decide Next Action (Based on Assessment):**
        a.  **Follow-up Needed?**
            *   **Condition:** Sentiment is Positive/Neutral, Content Assessment is Yes/Partially, AND `completeness_score < 80`, AND `follow_up_count < 3`.
            *   **Consider:** Check `structured_summary` for `unclear_details` or potential contradictions noted previously.
            *   **Action:** Output `DECISION: follow_up`. Generate a SPECIFIC, targeted follow-up question in `RESPONSE` focusing *only* on the missing information or clarification needed (referencing `Key points to cover` or `unclear_details`). Avoid broad requests. If sentiment was positive, gently explore factors (e.g., "听起来不错，是什么让您感觉好些了？").
        b.  **Move On (Sufficient Info):**
            *   **Condition:** Sentiment is Positive/Neutral, Content Assessment is Yes, AND `completeness_score >= 80`.
            *   **Action:** Output `DECISION: move_on`. Provide NO `RESPONSE` text (or only minimal acknowledgment like "好的。").
        c.  **Handle Negative/Abusive Sentiment:**
            *   **Condition:** Sentiment is Negative or Abusive.
            *   **Action:** Output `DECISION: de_escalate`. In `RESPONSE`, acknowledge emotion respectfully without condoning abuse. Use empathetic de-escalation. Examples: "听起来您似乎有些沮丧。能多告诉我一些您的感受吗？", "我理解您可能感到不满，但请避免使用攻击性言语，让我们专注于您的情况。" Do NOT thank for abusive responses.
        d.  **Handle Threatening Sentiment:**
            *   **Condition:** Sentiment is Threatening.
            *   **Action:** Output `DECISION: de_escalate` (or a specific `threat_detected` if needed). Prioritize safety. In `RESPONSE`, acknowledge seriously but avoid escalation. Example: "我注意到您提到了非常强烈的情绪。我想强调这里是安全的。" Consider safety protocols.
        e.  **Handle Irrelevant Response:**
            *   **Condition:** Content Assessment is 'No' AND `follow_up_count < 3`.
            *   **Check History:** Was the AI's *immediately preceding* turn a redirection attempt for this *same* question ID ({question.get('id', 'N/A')})?
            *   **If YES (Already Redirected):** Output `DECISION: follow_up`. `RESPONSE` should be a simple clarification attempt, e.g., "抱歉，我还是想了解一下关于[当前问题主题]的情况？"
            *   **If NO (First Redirect):** Output `DECISION: redirect`. Set `COMPLETENESS: 0`. In `RESPONSE`, gently redirect back to the `Current question` topic, e.g., "谢谢分享。为了更好地了解您的情况，我们能回到关于[当前问题主题]的问题上吗？" (Avoid repeating the full question text if possible, just the topic).

    4.  **Handle Maximum Follow-ups or Persistent Uncooperativeness:**
        *   **Condition:** `follow_up_count >= 3` (maximum reached) OR participant remains abusive/irrelevant after redirection/clarification attempts (e.g., `follow_up_count` is 2 or 3 and issues persist).
        *   **Action:** Output `DECISION: move_on`.
        *   **Reasoning:** State that the maximum follow-ups are reached or attempts to clarify/redirect were unsuccessful.
        *   **Response:** Provide NO `RESPONSE` text (or only minimal acknowledgment like "好的，我们继续下一个。").

    5.  **Handle Opening Turn:**
        *   **Condition:** This is the very first participant response in the interview (e.g., history length <= 2).
        *   **Action:**
            *   Ignore `Participant's response` content (unless abusive/threatening/refusing).
            *   Output `DECISION: move_on`.
            *   Set `COMPLETENESS: 100`.
            *   Provide NO `RESPONSE` text (or only minimal acknowledgment like "好的，我们开始吧。"). The first real question is handled externally.

---
Format your response EXACTLY as follows:

COMPLETENESS: [0-100 score]
DECISION: [move_on/follow_up/de_escalate/redirect]
KEY_POINTS_COVERED: [List of key points explicitly covered in the response (e.g., ["duration", "trigger"] or None)]
REASONING: [Concise justification for the decision. If 'follow_up', state missing info/clarification needed. If 'redirect', state why irrelevant. If 'move_on' due to Yes/No, state so. If 'move_on' due to max follow-ups, state so.]
RESPONSE: [
    IF DECISION is 'follow_up': Targeted follow-up question here.
    IF DECISION is 'redirect': Polite redirection statement here.
    IF DECISION is 'de_escalate': De-escalation statement here.
    IF DECISION is 'move_on': **LEAVE THIS EMPTY or provide ONLY a single, brief acknowledgment (e.g., "好的。"). The actual transition and next question text are handled externally.**
]
        """

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
        你的任务是接收一个标准的评估问题（“原问题”），并将其转化为一个适合在自然对话中提出的问题，同时严格遵守以下规则：

        **核心原则：忠于原文，仅做必要优化**

        1.  **优先保持原样:** 如果“原问题”本身已经足够清晰、自然，请 **直接输出原问题，不做任何修改**。这是首选。
        2.  **最小化修饰 (仅在必要时):**
            *   **何时修饰?** 仅当“原问题”显得过于生硬、缺乏上下文过渡，或直接提出可能引起不适时。
            *   **如何修饰?** 允许在“原问题”**前面**添加**不超过一句**的、**简短**的自然过渡语或表示理解/共情的话语（例如：“好的，接下来我们了解一下...” 或 “嗯，我明白这可能不容易谈起，关于...”）。
            *   **严格禁止:**
                *   **改变核心语义:** 不得改变问题的核心询问点。
                *   **删减/修改关键信息:** **绝对禁止**删除、简化或替换“原问题”中任何具体的症状描述、事件细节、时间范围等（例如，“会觉得有人追您”、“淋浴喷头后面有只手”、“听到有人让您去死”等必须原样保留）。
                *   **改变句子主体结构:** 原问题的主要句式和语序应保持不变。
                *   **过度修饰:** 避免添加冗余的客套话或解释。
        3.  **区分信息来源 (处理第三方信息):**
            *   如果“原问题”引用了病历、档案或非参与者亲口所述的信息，**切勿**使用“您说过”或“您提到过”。
            *   必须使用或添加类似“我了解到...”、“记录显示...”或“之前的信息提到...”的表述，明确信息来源。
            *   **但无论如何，信息来源之后的具体内容描述（如症状、事件）必须完整保留，不可删改。**
        4.  **避免重复提问 (检查历史):**
            *   **极其重要:** 在生成问题前，仔细检查**提供的** `对话历史` (通常是最近的对话片段)。
            *   **确认信息:** 确认参与者在**近期对话中**是否已经**明确**回答了与“原问题”相关的关键信息点（例如：症状的**首次出现时间**、**频率**、**持续时长**、**严重程度**、对特定情况的**态度**等）。
            *   **若已回答:** **绝对不要**再次提出相同的问题或询问已被确认的信息。在这种（罕见）情况下，如果脚本流程仍要求提出此问题，可能需要输出一个确认性的、不同的表述（但这超出了本 Prompt 的主要范围，优先假定脚本流程是合理的，避免重复是第一要务）。 *（注：此处的处理可能需要代码逻辑配合，Prompt 主要强调避免重复）*
        5.  **简洁与专业:** 避免使用过于晦涩的医疗术语（除非“原问题”本身包含且必须保留）。提问应直接、清晰。

        **输出格式要求:**

        *   **可选的前导语句:** 如果根据规则 #2 添加了过渡/共情语句，将其放在最前面（不超过一句）。
        *   **核心问题文本:** 紧接着输出处理后的“原问题”文本（保持完整性）。
        *   **无额外内容:** **绝对禁止**在最终输出的问题前后添加任何解释、说明、标注或诸如“这是您的问题：”之类的引导语。**直接输出最终要呈现给参与者的问题文本。**

        **示例：**

        *   **原问题:** 「您说您独处时，会觉得有人追您，还觉得淋浴喷头后面会有一只手伸出来，甚至听到有人让您去死。最近一周还有这种感觉吗？」
        *   **假设情景1 (无需修饰):** 直接输出：「您说您独处时，会觉得有人追您，还觉得淋浴喷头后面会有一只手伸出来，甚至听到有人让您去死。最近一周还有这种感觉吗？」
        *   **假设情景2 (需要轻微过渡):** 输出：「我知道这些经历可能让人非常不安。我们来确认一下：您说您独处时，会觉得有人追您，还觉得淋浴喷头后面会有一只手伸出来，甚至听到有人让您去死。最近这一周还有这种感觉吗？」 (仅加了一句过渡)
        *   **假设情景3 (涉及档案信息，参与者未亲口说过):**
            *   **原问题:** 「您说您感到生活没意思。」
            *   **需要修改为:** 「我了解到，您之前表达过感觉生活没什么意思。能具体谈谈这方面的情况吗？」 (使用“我了解到”，并保留核心内容)

        **再次强调:**

        *   **首要任务是保持原问题的完整性和准确性。**
        *   **仔细检查对话历史，避免重复提问。**
        *   **修饰是次要的，且必须极其克制。**

        **FINAL INSTRUCTION: Output ONLY the final question text intended for the participant. DO NOT include any explanations, apologies, or introductory phrases like 'Here is the rephrased question:'.**
        '''
            # Original didn't use .replace for placeholder, passed history in user message1 

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