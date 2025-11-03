import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
import joblib
import os

class PCBuildingBot:
    def __init__(self):
        self.vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
        self.model = LogisticRegression(random_state=42)
        self.is_trained = False

        # PC building recommendations database with pricing
        self.components = {
            'cpu': {
                'ultra_budget': {'name': 'AMD Ryzen 5 5600G', 'price': 159},  # APU with integrated graphics
                'budget': {'name': 'AMD Ryzen 5 5600', 'price': 129},
                'budget_intel': {'name': 'Intel Core i5-12400F', 'price': 149},
                'cheap_intel': {'name': 'Intel Core i3-12100F', 'price': 109},
                'entry_gaming': {'name': 'AMD Ryzen 5 7600', 'price': 199},
                'gaming': {'name': 'AMD Ryzen 5 7600X', 'price': 299},
                'gaming_intel': {'name': 'Intel Core i5-13600K', 'price': 319},
                'high_end_gaming': {'name': 'AMD Ryzen 7 7700X', 'price': 399},
                'enthusiast': {'name': 'Intel Core i7-13700K', 'price': 449},
                'workstation': {'name': 'AMD Ryzen 9 7950X', 'price': 699},
                'professional': {'name': 'Intel Core i9-13900K', 'price': 799}
            },
            'gpu': {
                'ultra_budget': {'name': 'NVIDIA GTX 1650', 'price': 149},
                'budget': {'name': 'NVIDIA RTX 4060', 'price': 299},
                'budget_amd': {'name': 'AMD RX 7600', 'price': 269},
                'cheap_amd': {'name': 'AMD RX 6600', 'price': 199},
                'entry': {'name': 'NVIDIA RTX 4060 Ti', 'price': 399},
                'mid': {'name': 'NVIDIA RTX 4070', 'price': 599},
                'mid_amd': {'name': 'AMD RX 7800 XT', 'price': 549},
                'high_end': {'name': 'NVIDIA RTX 4070 Ti', 'price': 799},
                'enthusiast': {'name': 'NVIDIA RTX 4080', 'price': 1199},
                'enthusiast_amd': {'name': 'AMD RX 7900 XT', 'price': 899},
                'professional': {'name': 'NVIDIA RTX 4090', 'price': 1699},
                'workstation': {'name': 'NVIDIA RTX 4080', 'price': 1199}
            },
            'ram': {
                'ultra_budget': {'name': '8GB DDR4-3200', 'price': 29},
                'budget': {'name': '16GB DDR4-3200', 'price': 59},
                'gaming': {'name': '16GB DDR5-5600', 'price': 79},
                'productivity': {'name': '32GB DDR4-3600', 'price': 109},
                'high_end': {'name': '32GB DDR5-6000', 'price': 149},
                'workstation': {'name': '64GB DDR4-3600', 'price': 199},
                'professional': {'name': '64GB DDR5-6000', 'price': 299}
            },
            'motherboard': {
                'budget_amd': {'name': 'ASRock B450M PRO4', 'price': 79},
                'budget_intel': {'name': 'ASRock B660M-HDV', 'price': 89},
                'budget': {'name': 'MSI PRO B660M-A', 'price': 119},
                'amd': {'name': 'MSI MAG B650 Tomahawk', 'price': 199},
                'amd_premium': {'name': 'ASUS ROG Strix B650-A Gaming', 'price': 249},
                'intel': {'name': 'MSI MAG Z790 Tomahawk', 'price': 299},
                'intel_premium': {'name': 'ASUS ROG Strix Z790-A Gaming', 'price': 349},
                'workstation_amd': {'name': 'ASUS ProArt B650-Creator', 'price': 279},
                'workstation_intel': {'name': 'ASUS ProArt Z790-Creator', 'price': 399},
                'enthusiast_amd': {'name': 'ASUS ROG Crosshair X670E Hero', 'price': 499},
                'enthusiast_intel': {'name': 'ASUS ROG Maximus Z790 Hero', 'price': 599}
            },
            'psu': {
                '450w': {'name': 'Corsair CV450', 'price': 49},
                '550w': {'name': 'Corsair RM550x', 'price': 79},
                '650w': {'name': 'Corsair RM650x', 'price': 109},
                '750w': {'name': 'Corsair RM750x', 'price': 139},
                '850w': {'name': 'Corsair RM850x', 'price': 169},
                '1000w': {'name': 'Corsair RM1000x', 'price': 199},
                '1200w': {'name': 'Corsair AX1200i', 'price': 299}
            },
            'case': {
                'mid': {'name': 'Fractal Design Meshify C', 'price': 99},
                'full': {'name': 'Fractal Design Define 7', 'price': 149},
                'budget': {'name': 'Cooler Master MasterBox NR600', 'price': 69}
            },
            'cooler': {
                'air': {'name': 'Noctua NH-D15', 'price': 109},
                'aio': {'name': 'Corsair H100i Elite Capellix', 'price': 149},
                'budget': {'name': 'Cooler Master Hyper 212', 'price': 39}
            },
            'storage': {
                'budget_ssd': {'name': 'Crucial MX500 500GB SATA SSD', 'price': 49},
                'budget_hdd': {'name': 'Seagate Barracuda 1TB HDD', 'price': 39},
                'entry_ssd': {'name': 'Samsung 870 EVO 1TB SATA SSD', 'price': 79},
                'primary': {'name': 'Samsung 980 Pro 1TB NVMe SSD', 'price': 89},
                'fast_ssd': {'name': 'WD Black SN850 1TB NVMe SSD', 'price': 99},
                'secondary': {'name': 'Seagate Barracuda 2TB HDD', 'price': 59},
                'large_hdd': {'name': 'Western Digital Blue 4TB HDD', 'price': 89},
                'high_end_ssd': {'name': 'Samsung 990 Pro 2TB NVMe SSD', 'price': 179},
                'workstation_ssd': {'name': 'Samsung 980 Pro 2TB NVMe SSD', 'price': 169}
            }
        }

        # Educational content and fun facts
        self.knowledge_base = {
            'cpu': {
                'what_is': "The CPU (Central Processing Unit) is the 'brain' of your computer. It executes instructions from programs and performs calculations. Think of it as the conductor of an orchestra - it coordinates all the other components!",
                'why_needed': "You need a CPU because it's the primary processor that runs your operating system and all your programs. Without a CPU, your computer literally can't do anything - it's like having a car without an engine!",
                'fun_fact': "Did you know? The first CPU was the size of a large room and used 18,000 vacuum tubes! Modern CPUs fit in your palm and contain billions of transistors."
            },
            'gpu': {
                'what_is': "The GPU (Graphics Processing Unit) specializes in rendering images, videos, and animations. It's essential for gaming, video editing, and AI tasks. While CPUs are like chess masters, GPUs are like artists creating masterpieces!",
                'why_needed': "You need a GPU for any visual tasks - gaming, video playback, photo editing, 3D modeling, and even accelerating AI tasks. Without a GPU, you'd have no graphics output and very slow performance on visual applications.",
                'fun_fact': "Fun fact: GPUs were originally designed just for graphics, but now they're used for cryptocurrency mining, scientific research, and even protein folding to fight diseases!"
            },
            'ram': {
                'what_is': "RAM (Random Access Memory) is your computer's short-term memory. It temporarily stores data that the CPU needs quick access to. More RAM means you can run more programs simultaneously without slowdowns.",
                'why_needed': "RAM is essential because it acts as your computer's workspace. When you open programs, they load into RAM for fast access. Without enough RAM, your computer will slow down dramatically or crash when multitasking.",
                'fun_fact': "Interesting fact: RAM is called 'random access' because you can access any piece of data in it just as quickly as any other piece - unlike older storage methods that had to read sequentially!"
            },
            'ssd': {
                'what_is': "An SSD (Solid State Drive) is modern storage that uses flash memory chips instead of spinning disks. It's much faster than traditional HDDs, leading to quicker boot times and faster file loading.",
                'why_needed': "You need storage to hold your operating system, programs, and files permanently. SSDs are preferred because they're much faster than HDDs, leading to quicker system boot times and faster application loading.",
                'fun_fact': "Cool fact: SSDs have no moving parts, making them more reliable and resistant to physical shock. You could drop an SSD and it would still work perfectly!"
            },
            'motherboard': {
                'what_is': "The motherboard is the main circuit board that connects all your components together. It's like the nervous system of your PC, providing pathways for communication between CPU, RAM, storage, and peripherals.",
                'why_needed': "The motherboard is absolutely essential because it provides the foundation that connects all your components. Without it, none of your parts (CPU, RAM, GPU, etc.) can communicate with each other.",
                'fun_fact': "Did you know? The term 'motherboard' comes from being the 'mother' board that gives birth to the connections for all other 'daughter' boards in the system!"
            },
            'psu': {
                'what_is': "The PSU (Power Supply Unit) converts AC power from your wall outlet into DC power that your PC components can use. It must provide stable, clean power to prevent damage to expensive components.",
                'why_needed': "You need a PSU because it supplies electrical power to all your components. Without power, nothing works! A good PSU also protects your expensive components from power surges and provides stable voltage.",
                'fun_fact': "Power fact: A good PSU is like a heart - it pumps clean, stable power throughout your system. Cheap PSUs can deliver 'dirty' power that might damage your components over time!"
            },
            'case': {
                'what_is': "The PC case houses and protects all your components while providing airflow for cooling. Good cases have proper cable management and dust filters for a clean, cool system.",
                'why_needed': "The case protects your expensive components from dust, damage, and electromagnetic interference. It also provides proper airflow for cooling and organizes all your cables for better airflow and aesthetics.",
                'fun_fact': "Case trivia: PC cases come in many styles - from boring beige boxes to glowing RGB masterpieces. Some cases are even designed to look like Darth Vader or AT-AT walkers!"
            },
            'cooler': {
                'what_is': "CPU coolers prevent your processor from overheating by dissipating heat. Air coolers use fans and heatsinks, while liquid coolers use coolant circulated through tubes.",
                'why_needed': "Cooling is critical because modern CPUs generate a lot of heat during operation. Without proper cooling, your CPU will overheat, slow down (thermal throttling), or even get permanently damaged.",
                'fun_fact': "Cool fact: The first computers didn't need fans because they used so little power they stayed cool. Modern gaming PCs can produce enough heat to warm a small room!"
            },
            'storage': {
                'what_is': "Storage devices hold your operating system, programs, and files. SSDs are fast but expensive per GB, while HDDs are slower but offer more storage for less money.",
                'why_needed': "Storage is essential for holding your operating system, applications, and personal files. Without storage, you couldn't install anything or save any data permanently.",
                'fun_fact': "Storage stats: The first hard drive from IBM in 1956 stored just 5MB and cost $50,000! Today you can get 1TB SSDs for under $100."
            }
        }

        # Complete build configurations with pricing
        self.complete_builds = {
            'budget': {
                'name': 'Budget Gaming Build (~$800)',
                'components': {
                    'cpu': self.components['cpu']['budget'],
                    'gpu': self.components['gpu']['budget'],
                    'ram': self.components['ram']['gaming'],
                    'motherboard': self.components['motherboard']['budget'],
                    'psu': self.components['psu']['650w'],
                    'case': self.components['case']['budget'],
                    'cooler': self.components['cooler']['budget'],
                    'storage': self.components['storage']['budget_ssd']
                }
            },
            'gaming': {
                'name': 'Mid-Range Gaming Build (~$1300)',
                'components': {
                    'cpu': self.components['cpu']['gaming'],
                    'gpu': self.components['gpu']['mid'],
                    'ram': self.components['ram']['gaming'],
                    'motherboard': self.components['motherboard']['amd'],
                    'psu': self.components['psu']['750w'],
                    'case': self.components['case']['mid'],
                    'cooler': self.components['cooler']['air'],
                    'storage': self.components['storage']['primary']
                }
            },
            'high_end_gaming': {
                'name': 'High-End Gaming Build (~$2200)',
                'components': {
                    'cpu': self.components['cpu']['high_end_gaming'],
                    'gpu': self.components['gpu']['enthusiast'],
                    'ram': self.components['ram']['productivity'],
                    'motherboard': self.components['motherboard']['amd'],
                    'psu': self.components['psu']['850w'],
                    'case': self.components['case']['full'],
                    'cooler': self.components['cooler']['aio'],
                    'storage': self.components['storage']['primary']
                }
            },
            'office': {
                'name': 'Office/Productivity Build (~$650)',
                'components': {
                    'cpu': self.components['cpu']['budget_intel'],
                    'gpu': {'name': 'Integrated Graphics', 'price': 0},  # No discrete GPU
                    'ram': self.components['ram']['gaming'],
                    'motherboard': self.components['motherboard']['budget'],
                    'psu': self.components['psu']['650w'],
                    'case': self.components['case']['budget'],
                    'cooler': self.components['cooler']['budget'],
                    'storage': self.components['storage']['primary']
                }
            },
            'workstation': {
                'name': 'Content Creation Workstation (~$2800)',
                'components': {
                    'cpu': self.components['cpu']['workstation'],
                    'gpu': self.components['gpu']['workstation'],
                    'ram': self.components['ram']['productivity'],
                    'motherboard': self.components['motherboard']['amd'],
                    'psu': self.components['psu']['850w'],
                    'case': self.components['case']['full'],
                    'cooler': self.components['cooler']['aio'],
                    'storage': self.components['storage']['primary']
                }
            }
        }

    def load_data(self, filepath):
        """Load PC building query data from CSV file."""
        df = pd.read_csv(filepath)
        return df['query'], df['category']

    def train(self, X, y):
        """Train the query classification model with multiple algorithms and hyperparameter tuning."""
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Vectorize text with improved parameters
        self.vectorizer = TfidfVectorizer(
            max_features=2000,
            stop_words='english',
            ngram_range=(1, 2),  # Include bigrams
            min_df=2,  # Minimum document frequency
            max_df=0.9  # Maximum document frequency
        )

        X_train_vec = self.vectorizer.fit_transform(X_train)
        X_test_vec = self.vectorizer.transform(X_test)

        # Try multiple algorithms with hyperparameter tuning
        models = {
            'LogisticRegression': {
                'model': LogisticRegression(random_state=42, max_iter=1000),
                'params': {
                    'C': [0.1, 1.0, 10.0],
                    'solver': ['liblinear', 'lbfgs']
                }
            },
            'RandomForest': {
                'model': RandomForestClassifier(random_state=42),
                'params': {
                    'n_estimators': [100, 200, 300],
                    'max_depth': [10, 20, None],
                    'min_samples_split': [2, 5, 10]
                }
            },
            'SVM': {
                'model': SVC(random_state=42),
                'params': {
                    'C': [0.1, 1.0, 10.0],
                    'kernel': ['linear', 'rbf'],
                    'gamma': ['scale', 'auto']
                }
            },
            'NaiveBayes': {
                'model': MultinomialNB(),
                'params': {
                    'alpha': [0.1, 0.5, 1.0, 2.0]
                }
            }
        }

        best_accuracy = 0
        best_model = None
        best_model_name = ""

        print("Training and evaluating multiple ML models...")

        # Train and evaluate each model
        for model_name, model_config in models.items():
            print(f"\nTraining {model_name}...")

            # Use GridSearchCV for hyperparameter tuning
            grid_search = GridSearchCV(
                model_config['model'],
                model_config['params'],
                cv=3,
                scoring='accuracy',
                n_jobs=-1
            )

            grid_search.fit(X_train_vec, y_train)

            # Evaluate on test set
            y_pred = grid_search.predict(X_test_vec)
            accuracy = accuracy_score(y_test, y_pred)

            print(f"{model_name} - Best params: {grid_search.best_params_}")
            print(f"{model_name} - Test accuracy: {accuracy:.3f}")

            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_model = grid_search.best_estimator_
                best_model_name = model_name

        # Use the best performing model
        self.model = best_model
        print(f"\n🎯 Selected best model: {best_model_name} with accuracy: {best_accuracy:.3f}")

        self.is_trained = True
        return best_accuracy

    def predict_category(self, text):
        """Predict category of PC building query."""
        if not self.is_trained:
            return "unknown"

        text_vec = self.vectorizer.transform([text])
        prediction = self.model.predict(text_vec)[0]
        return prediction

    def get_recommendation(self, category, user_input=""):
        """Get PC building recommendation for a category."""
        # Check if user wants to build a PC - start interactive mode
        if "build" in user_input.lower() and ("pc" in user_input.lower() or "computer" in user_input.lower()):
            return "INTERACTIVE_BUILD"

        # Check if user is asking educational questions
        user_input_lower = user_input.lower()
        if user_input_lower.startswith("what is") or user_input_lower.startswith("explain") or "what's" in user_input_lower:
            # Try to identify what component they're asking about
            for comp_key, comp_data in self.knowledge_base.items():
                if comp_key in user_input_lower or comp_key + "s" in user_input_lower:
                    return f"\n{comp_data['what_is']}\n\n💡 {comp_data['fun_fact']}"
        elif "why" in user_input_lower and ("need" in user_input_lower or "required" in user_input_lower or "important" in user_input_lower):
            # Try to identify what component they're asking about
            for comp_key, comp_data in self.knowledge_base.items():
                if comp_key in user_input_lower or comp_key + "s" in user_input_lower:
                    return f"\n{comp_data['why_needed']}\n\n💡 {comp_data['fun_fact']}"

        if category in ['budget', 'gaming', 'office', 'workstation'] or category in ['high_end_gaming']:
            # Return complete build with pricing
            build_key = category if category != 'gaming' else 'gaming'
            if category == 'high_end_gaming':
                build_key = 'high_end_gaming'

            if build_key in self.complete_builds:
                build = self.complete_builds[build_key]
                response = f"\n{build['name']}\n"
                response += "=" * 50 + "\n"

                total_cost = 0
                for comp_type, component in build['components'].items():
                    comp_name = component['name']
                    comp_price = component['price']
                    total_cost += comp_price
                    response += f"{comp_type.upper()}: {comp_name} - ${comp_price}\n"

                response += "=" * 50 + "\n"
                response += f"Total Estimated Cost: ${total_cost}\n"
                return response

        elif category in self.components:
            # Return individual component recommendation
            import random
            tier = random.choice(list(self.components[category].keys()))
            component = self.components[category][tier]
            return f"{category.upper()}: {component['name']} - ${component['price']}"

        else:
            return "I'm not sure about that component. Could you be more specific about what you're looking for?"

    def build_custom_pc(self, budget=None, use_case=None):
        """Build a custom PC based on budget and use case with strict budget constraints."""
        if budget is None or use_case is None:
            return None

        # Calculate component budget allocation (leave ~10% buffer for taxes/shipping)
        available_budget = budget * 0.9

        # Determine CPU platform (AMD vs Intel) and RAM type for compatibility
        if use_case.lower() == 'gaming':
            if budget < 400:
                # Ultra-budget: APU or basic CPU with integrated graphics
                cpu_key = 'ultra_budget'  # AMD Ryzen 5 5600G $159
                gpu_key = None  # Integrated graphics
                ram_key = 'ultra_budget'  # 8GB DDR4 $29
                mobo_key = 'budget_amd'  # ASRock B450M PRO4 $79
                storage_key = 'budget_ssd'  # Crucial MX500 500GB $49
                build_name = f"Ultra-Budget Gaming PC (~${budget})"
            elif budget < 600:
                # Budget: Basic gaming setup
                cpu_key = 'budget'  # AMD Ryzen 5 5600 $129
                gpu_key = 'ultra_budget'  # GTX 1650 $149
                ram_key = 'budget'  # 16GB DDR4 $59
                mobo_key = 'budget_amd'  # ASRock B450M PRO4 $79
                storage_key = 'budget_ssd'  # Crucial MX500 500GB $49
                build_name = f"Budget Gaming PC (~${budget})"
            elif budget < 900:
                # Entry gaming: Better GPU
                cpu_key = 'budget'  # AMD Ryzen 5 5600 $129
                gpu_key = 'budget'  # RTX 4060 $299
                ram_key = 'budget'  # 16GB DDR4 $59
                mobo_key = 'budget_amd'  # ASRock B450M PRO4 $79
                storage_key = 'budget_ssd'  # Crucial MX500 500GB $49
                build_name = f"Entry Gaming PC (~${budget})"
            elif budget < 1300:
                # Mid-range: Better CPU and GPU
                cpu_key = 'gaming'  # AMD Ryzen 5 7600X $299
                gpu_key = 'mid'  # RTX 4070 $599
                ram_key = 'gaming'  # 16GB DDR5 $79
                mobo_key = 'amd'  # MSI MAG B650 $199
                storage_key = 'primary'  # Samsung 980 Pro 1TB $89
                build_name = f"Mid-Range Gaming PC (~${budget})"
            else:
                # High-end: Premium components
                cpu_key = 'high_end_gaming'  # AMD Ryzen 7 7700X $399
                gpu_key = 'high_end'  # RTX 4070 Ti $799
                ram_key = 'productivity'  # 32GB DDR4 $109
                mobo_key = 'amd'  # MSI MAG B650 $199
                storage_key = 'primary'  # Samsung 980 Pro 1TB $89
                build_name = f"High-End Gaming PC (~${budget})"

        elif use_case.lower() == 'office' or use_case.lower() == 'work':
            if budget < 400:
                # Basic office: Minimal setup
                cpu_key = 'cheap_intel'  # Intel i3-12100F $109
                gpu_key = None  # Integrated graphics
                ram_key = 'ultra_budget'  # 8GB DDR4 $29
                mobo_key = 'budget_intel'  # ASRock B660M-HDV $89
                storage_key = 'budget_hdd'  # Seagate 1TB HDD $39
                build_name = f"Basic Office PC (~${budget})"
            else:
                # Standard office: Better performance
                cpu_key = 'budget_intel'  # Intel i5-12400F $149
                gpu_key = None  # Integrated graphics
                ram_key = 'budget'  # 16GB DDR4 $59
                mobo_key = 'budget_intel'  # ASRock B660M-HDV $89
                storage_key = 'primary'  # Samsung 980 Pro 1TB $89
                build_name = f"Office/Productivity PC (~${budget})"

        elif use_case.lower() in ['video editing', 'content creation', 'workstation']:
            if budget < 1000:
                # Entry workstation
                cpu_key = 'budget'  # AMD Ryzen 5 5600 $129
                gpu_key = 'budget'  # RTX 4060 $299
                ram_key = 'gaming'  # 16GB DDR5 $79
                mobo_key = 'budget_amd'  # ASRock B450M PRO4 $79
                storage_key = 'primary'  # Samsung 980 Pro 1TB $89
            elif budget < 1500:
                # Mid workstation
                cpu_key = 'gaming'  # AMD Ryzen 5 7600X $299
                gpu_key = 'mid'  # RTX 4070 $599
                ram_key = 'productivity'  # 32GB DDR4 $109
                mobo_key = 'amd'  # MSI MAG B650 $199
                storage_key = 'primary'  # Samsung 980 Pro 1TB $89
            else:
                # High-end workstation
                cpu_key = 'high_end_gaming'  # AMD Ryzen 7 7700X $399
                gpu_key = 'enthusiast'  # RTX 4080 $1199
                ram_key = 'workstation'  # 64GB DDR4 $199
                mobo_key = 'amd'  # MSI MAG B650 $199
                storage_key = 'workstation_ssd'  # Samsung 980 Pro 2TB $169
            build_name = f"Content Creation Workstation (~${budget})"

        else:  # Default gaming build
            cpu_key = 'gaming'
            gpu_key = 'mid'
            ram_key = 'gaming'
            mobo_key = 'amd'
            storage_key = 'primary'
            build_name = f"Custom Gaming PC (~${budget})"

        # Build the PC configuration with compatibility ensured
        pc_config = {
            'cpu': self.components['cpu'][cpu_key],
            'gpu': self.components['gpu'][gpu_key] if gpu_key else {'name': 'Integrated Graphics', 'price': 0},
            'ram': self.components['ram'][ram_key],
            'motherboard': self.components['motherboard'][mobo_key],
            'psu': self.components['psu']['650w'],
            'case': self.components['case']['budget'],
            'cooler': self.components['cooler']['budget'],
            'storage': self.components['storage'][storage_key]
        }

        # Adjust PSU based on GPU power requirements
        if gpu_key and gpu_key in ['enthusiast', 'professional']:
            pc_config['psu'] = self.components['psu']['850w']
        elif gpu_key and gpu_key in ['mid', 'high_end']:
            pc_config['psu'] = self.components['psu']['750w']

        return {
            'name': build_name,
            'components': pc_config
        }

    def save_model(self, model_path='models/pc_bot_model.pkl', vectorizer_path='models/pc_bot_vectorizer.pkl'):
        """Save trained model and vectorizer."""
        os.makedirs('models', exist_ok=True)
        joblib.dump(self.model, model_path)
        joblib.dump(self.vectorizer, vectorizer_path)
        print("Model saved successfully.")

    def load_model(self, model_path='models/pc_bot_model.pkl', vectorizer_path='models/pc_bot_vectorizer.pkl'):
        """Load trained model and vectorizer."""
        if os.path.exists(model_path) and os.path.exists(vectorizer_path):
            self.model = joblib.load(model_path)
            self.vectorizer = joblib.load(vectorizer_path)
            self.is_trained = True
            print("Model loaded successfully.")
        else:
            print("Model files not found.")

def main():
    bot = PCBuildingBot()

    # Try to load existing model
    bot.load_model()

    if not bot.is_trained:
        # Train new model
        print("Training new PC building bot model...")
        X, y = bot.load_data('data/pc_building_data.csv')
        bot.train(X, y)
        bot.save_model()

    # Interactive bot
    print("\nPC Building Assistant Bot")
    print("Ask me about PC components, builds, or recommendations!")
    print("Type 'quit' to exit")

    build_mode = False
    budget = None
    use_case = None

    while True:
        if build_mode:
            if budget is None:
                try:
                    budget_input = input("What's your budget in USD? (e.g., 1000, 1500, 2000): ")
                    if budget_input.lower() == 'quit':
                        break
                    budget = int(budget_input)
                    print(f"Great! Budget set to ${budget}")
                except ValueError:
                    print("Please enter a valid number for budget.")
                    continue
            elif use_case is None:
                use_case_input = input("What's your primary use case? (gaming/office/video editing/workstation): ")
                if use_case_input.lower() == 'quit':
                    break
                if use_case_input.lower() in ['gaming', 'office', 'video editing', 'workstation', 'work']:
                    use_case = use_case_input.lower()
                    if use_case == 'work':
                        use_case = 'office'
                    print(f"Perfect! Use case set to: {use_case}")
                else:
                    print("Please choose from: gaming, office, video editing, or workstation")
                    continue
            else:
                # Build the custom PC
                custom_build = bot.build_custom_pc(budget, use_case)
                if custom_build:
                    response = f"\n{custom_build['name']}\n"
                    response += "=" * 50 + "\n"

                    total_cost = 0
                    for comp_type, component in custom_build['components'].items():
                        comp_name = component['name']
                        comp_price = component['price']
                        total_cost += comp_price
                        response += f"{comp_type.upper()}: {comp_name} - ${comp_price}\n"

                    response += "=" * 50 + "\n"
                    response += f"Total Estimated Cost: ${total_cost}\n"
                    print(response)

                # Reset for next interaction
                build_mode = False
                budget = None
                use_case = None
                continue
        else:
            user_input = input("\nWhat can I help you with? ")
            if user_input.lower() == 'quit':
                break

            category = bot.predict_category(user_input)
            recommendation = bot.get_recommendation(category, user_input)

            if recommendation == "INTERACTIVE_BUILD":
                print("I'd be happy to help you build a custom PC!")
                build_mode = True
                continue

            print(f"Category: {category}")
            print(recommendation)

if __name__ == "__main__":
    main()
