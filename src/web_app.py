from flask import Flask, render_template, request, jsonify
import sys
import os
import logging
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from pc_bot import PCBuildingBot

# Enable logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

app = Flask(__name__)
bot = PCBuildingBot()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    user_input = request.json.get('message', '')

    if not user_input:
        return jsonify({'response': 'Please enter a message.'})

    # Check if user wants to build a PC
    if "build" in user_input.lower() and ("pc" in user_input.lower() or "computer" in user_input.lower()):
        return jsonify({
            'response': 'I\'d be happy to help you build a custom PC! Please provide your budget and use case.',
            'show_build_form': True
        })

    # Check educational questions
    user_input_lower = user_input.lower()
    if user_input_lower.startswith("what is") or user_input_lower.startswith("explain") or "what's" in user_input_lower:
        for comp_key, comp_data in bot.knowledge_base.items():
            if comp_key in user_input_lower or comp_key + "s" in user_input_lower:
                return jsonify({
                    'response': f"{comp_data['what_is']}\n\n💡 {comp_data['fun_fact']}",
                    'category': 'educational'
                })
    elif "why" in user_input_lower and ("need" in user_input_lower or "required" in user_input_lower or "important" in user_input_lower):
        for comp_key, comp_data in bot.knowledge_base.items():
            if comp_key in user_input_lower or comp_key + "s" in user_input_lower:
                return jsonify({
                    'response': f"{comp_data['why_needed']}\n\n💡 {comp_data['fun_fact']}",
                    'category': 'educational'
                })

    # Get bot recommendation
    category = bot.predict_category(user_input)
    recommendation = bot.get_recommendation(category, user_input)

    return jsonify({
        'response': recommendation,
        'category': category
    })

@app.route('/build_pc', methods=['POST'])
def build_pc():
    logger.info(f"Received build request: {request.json}")
    budget = request.json.get('budget')
    use_case = request.json.get('useCase')  # JavaScript sends camelCase

    if not budget or not use_case:
        logger.error("Missing budget or use_case")
        return jsonify({'error': 'Budget and use case are required.'})

    try:
        budget = int(budget)
        logger.info(f"Building PC for budget ${budget}, use case: {use_case}")
        custom_build = bot.build_custom_pc(budget, use_case)
        logger.info(f"Build result: {custom_build}")

        if custom_build:
            components = custom_build['components']
            total_cost = sum(comp['price'] for comp in components.values())

            # Format response
            response = f"{custom_build['name']}\n"
            response += "=" * 50 + "\n"

            for comp_type, component in components.items():
                response += f"{comp_type.upper()}: {component['name']} - ${component['price']}\n"

            response += "=" * 50 + "\n"
            response += f"Total Estimated Cost: ${total_cost}"

            logger.info(f"Returning successful build with total cost: ${total_cost}")
            return jsonify({
                'success': True,
                'build': response,
                'total_cost': total_cost,
                'budget': budget
            })
        else:
            logger.error(f"Could not generate build for budget ${budget} and use case {use_case}")
            return jsonify({'error': f'Could not generate build for budget ${budget} and use case "{use_case}". Please try a higher budget or different use case.'})

    except Exception as e:
        logger.error(f"Exception in build_pc: {str(e)}")
        return jsonify({'error': f'An error occurred: {str(e)}'})

if __name__ == '__main__':
    app.run(debug=True)
