import sys
from pathlib import Path
import questionary
from questionary import Choice
from loguru import logger


def setup_paths():
    current_dir = Path(__file__).resolve().parent
    #would be usefull for many folders
    sys.path.extend([
        str(current_dir / 'migration'),
        str(current_dir / 'rating'),
        str(current_dir / 'salary'),
        str(current_dir / 'scripts'),
    ])


def setup_logger():
    logger.remove()
    
    logger.add("./log/logging.log",
               format="{time:YYYY-MM-DD HH:mm:ss} | {level:<8} | {module:<30} | {line:<4} | {message}",
               level="INFO",
               rotation="11 MB",
               compression="zip")

    logger.add(lambda msg: print(msg, end=""),
               format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
                      "<level>{level:<8}</level> | "
                      "<cyan>{module:<30}</cyan> | "
                      "<magenta>{line:<4}</magenta> | "
                      "{message}",
               level="INFO",
               colorize=True)


def run_clean_data():
    try:
        from cleaning_data import clean_data
        clean_data()
    except ImportError as e:
        logger.error(f"import error: {e}")
    except Exception as e:
        logger.error(f"module error: {e}")


def run_scaling():
    try:
        from scaling import scaling
        scaling()
    except ImportError as e:
        logger.error(f"import error: {e}")
    except Exception as e:
        logger.error(f"module error: {e}")


def run_compare_models():
    try:
        from compare_models import compare_models
        compare_models()
    except ImportError as e:
        logger.error(f"import error: {e}")
    except Exception as e:
        logger.error(f"module error: {e}")


def run_apply_model():
    try:
        from apply_model import apply_model
        apply_model()
    except ImportError as e:
        logger.error(f"import error: {e}")
    except Exception as e:
        logger.error(f"module error: {e}")

        
def get_module():
    result = questionary.select(
        "Choose a script to launch:",
        choices=[
            Choice("00) Clean data", "clean_data"), #we can clean data only in order to check it. however. cleaning data is already included in scaling, so we can skip it here
            Choice("1) Clean data + scale", "scaling"),
            Choice("2) Compare models", "compare_models"),
            Choice("3) Apply model", "apply_model"),
            Choice("99) Quit", "exit")
        ],
        qmark="⚙️ ",
        pointer="✅ ",
    ).ask()
    return result


def main():
    #setup_paths()
    setup_logger()
    module = get_module()
    
    if module == "clean_data":
        run_clean_data()
    elif module == "scaling":
        run_scaling()
    elif module == "compare_models":
        run_compare_models()
    elif module == "apply_model":
        run_apply_model()                         
    elif module == "exit":
        logger.info("Goodbye!")
        sys.exit()


if __name__ == '__main__':
    main()
