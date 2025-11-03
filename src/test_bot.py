from pc_bot import PCBuildingBot

def test_bot():
    bot = PCBuildingBot()

    # Load or train model
    bot.load_model()

    if not bot.is_trained:
        print("Training new PC building bot model...")
        X, y = bot.load_data('data/pc_building_data.csv')
        accuracy = bot.train(X, y)
        print(f"Training accuracy: {accuracy:.2f}")
        bot.save_model()

    # Test cases
    test_queries = [
        "What CPU should I get for gaming?",
        "Recommend a good graphics card",
        "How much RAM do I need?",
        "I need a good power supply",
        "What's good for video editing?",
        "Budget PC build under 500",
        "High-end gaming PC build",
        "Build me an office computer",
        "I want a workstation for content creation",
        "Best CPU for professional work",
        "High-end GPU recommendation"
    ]

    print("\nTesting PC Building Assistant Bot:")
    print("-" * 60)

    for query in test_queries:
        category = bot.predict_category(query)
        recommendation = bot.get_recommendation(category)
        print(f"Query: '{query}'")
        print(f"Category: {category}")
        print(recommendation)
        print("-" * 60)

if __name__ == "__main__":
    test_bot()
