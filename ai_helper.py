# ai_helper.py
from openai import OpenAI
import os
import logging
import re
from utils import log_sensitive
import time

# AI Model Configurations
AI_MODELS = {
    "deepseek-v3": {
        "api_key": os.getenv("DEEPSEEK_API_KEY"),
        "base_url": "https://api.deepseek.com",
        "model": "deepseek-chat",
    },
    "ARK-deepseek-v3": {
        "api_key": os.getenv("ARK_API_KEY"),
        "base_url": "https://ark.cn-beijing.volces.com/api/v3",
        "model": "deepseek-v3-241226",
    },
    "doubao-1-5-lite-32k": {
        "api_key": os.getenv("ARK_API_KEY"),
        "base_url": "https://ark.cn-beijing.volces.com/api/v3",
        "model": "doubao-1-5-lite-32k-250115",
    },
    "openai-gpt-4o": {
        "api_key": os.getenv("OPENAI_API_KEY"),
        "base_url": "https://api.openai.com/v1",
        "model": "gpt-4o",
    },
    "anthropic": {
        "api_key": os.getenv("ANTHROPIC_API_KEY"),
        "base_url": "https://api.anthropic.com",
        "model": "claude-3-sonnet-20240229",
    }
}
# Default model configuration
CURRENT_MODEL = "ARK-deepseek-v3"

def get_ai_action(player, game_state):
    """
    Returns the action for an AI player based on the current game state.
    The response from LLMs API will include a chain-of-thought (under 'Thought:') and 
    a final decision (under 'Action:'). The function extracts and returns the action.
    """
    # Get model configuration
    model_config = AI_MODELS[CURRENT_MODEL]
    
    # Initialize AI client
    client = OpenAI(
        api_key=model_config["api_key"],
        base_url=model_config["base_url"]
    )
    
    # Build the system message (instructions)
    system_prompt = (
        "You are a professional poker AI. Analyze the game situation and output your "
        "chain-of-thought followed by your final decision. Your response must have two "
        "sections: 'Thought:' for your reasoning, and 'Action:' for your final decision. "
        "Allowed actions are: fold, call, check, raise [amount], or all-in. "
        "Only output one decision."
    )
    
    # Build the user prompt with all available information
    user_prompt = (
        f"Your position: {game_state.get('your_position')}\n"
        f"Your hole cards: {player.hole_cards}\n"
        f"Community cards: {game_state.get('community_cards', [])}\n"
        f"Current pot: {game_state.get('pot')}\n"
        f"Your current bet in this round: {player.current_bet}\n"
        f"Current highest bet in this round: {game_state.get('current_bet')}\n"
        f"Game round: {game_state.get('round')}\n"
        "\nOpponent Information:\n"
    )

    # Add opponent information in a structured way
    for opp in game_state.get('opponents', []):
        user_prompt += (
            f"- {opp['name']} ({opp['position']}): "
            f"Chips={opp['chips']}, Bet={opp['current_bet']}"
        )
        if opp['action_history']:
            user_prompt += "\n  Action History:\n"
            for action in opp['action_history']:
                if action['amount']:
                    user_prompt += f"    {action['round']}: {action['action']} {action['amount']}\n"
                else:
                    user_prompt += f"    {action['round']}: {action['action']}\n"
        user_prompt += "\n"

    user_prompt += (
        "\nPlease analyze the situation and then output your decision in the following format:\n"
        "Thought: <your reasoning>\n"
        "Action: <your decision>\n"
        "Ensure your output includes both sections. "
        "Action in the format of 'fold', 'call', 'check', 'raise [amount]', or 'all-in'."
    )

    # Get conversation history for this AI player
    messages = [{"role": "system", "content": system_prompt}]
    
    # Add previous conversation history if it exists
    if hasattr(player, 'conversation_history'):
        messages.extend(player.conversation_history)
    else:
        player.conversation_history = []
    
    # Add current prompt
    messages.append({"role": "user", "content": user_prompt})
    
    # Call AI API using ChatCompletion
    response = client.chat.completions.create(
        model=model_config["model"],
        messages=messages,
        stream=False
    )
    
    ai_response = response.choices[0].message.content.strip()
    
    # Log the complete AI interaction
    log_sensitive("\nAI INTERACTION", {
        "player": player.name,
        "\nprompt": user_prompt,
        "\nresponse": ai_response
    })
    print(f"AI response: {ai_response}")
    # Store the conversation history (keep last 5 exchanges to maintain context without too much token usage)
    player.conversation_history.append({"role": "user", "content": user_prompt})
    player.conversation_history.append({"role": "assistant", "content": ai_response})
    if len(player.conversation_history) > 10:  # Keep last 5 exchanges (10 messages)
        player.conversation_history = player.conversation_history[-10:]
    
    # Extract action from response using regex
    action_match = re.search(r"Action:\s*(.+?)(?:\n|$)", ai_response)
    if not action_match:
        raise ValueError(f"Invalid AI response format: {ai_response}")
    
    return action_match.group(1).strip()



def test_api_availability(model_name=None):
    """
    Test the availability and latency of AI APIs.
    
    Args:
        model_name (str, optional): Specific model to test. If None, test all configured models.
    
    Returns:
        dict: Dictionary containing test results for each model with status and latency
    """
    def test_single_model(model, config):
        print(f"Start test {model} \n{config}")
                         
        if not config.get("api_key"):
            return {"status": "error", "message": "API key not found", "latency": None}
            
        try:
            client = OpenAI(api_key=config["api_key"], base_url=config["base_url"])
            start_time = time.time()
            
            response = client.chat.completions.create(
                model=config["model"],
                messages=[{"role": "user", "content": "Test message"}],
                stream=False
            )
            
            latency = round((time.time() - start_time) * 1000)
            
            return {
                "status": "success" if response.choices else "error",
                "message": "API is working" if response.choices else "Invalid response",
                "latency": latency
            }
            
        except Exception as e:
            return {"status": "error", "message": str(e), "latency": None}

    models_to_test = [model_name] if model_name else AI_MODELS.keys()
    results = {}
    
    for model in models_to_test:
        if model not in AI_MODELS:
            results[model] = {"status": "error", "message": "Model not configured", "latency": None}
            continue
            
        results[model] = test_single_model(model, AI_MODELS[model])
        print(f"{model}: {results[model]}")
    
    return results


if __name__ == "__main__":
    test_api_availability("doubao-1-5-lite-32k")