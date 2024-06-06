from src.pipelines import get_pipeline
from src.utils import setup_config


def main(config):
    pipeline = get_pipeline(config)
    pipeline.run()

if __name__ == '__main__':
    config = setup_config()
    # setup_dir(config)
    
    main(config)

