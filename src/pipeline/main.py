from src.pipeline.pipeline import SegmentationPipeline
from src.config.constant import *


def main():
    pipeline = SegmentationPipeline(
        meta_data=META_DATA,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        batch_size=BATCH_SIZE,
        num_sequences=NUM_SEQUENCES,
        emb_dim=EMB_DIM,
        lr=LR,
        weight_decay=WEIGHT_DECAY,
        mode=MODE,
        factor=FACTOR,
        patience=PATIENCE,
        epochs=EPOCHS,
        checkpoint_path=CHECKPOINT_PATH,
        history_path=HISTORY_PLOT_PATH,
        output_dir=OUTPUT_DIR,
    )

    history, results = pipeline.run_full_pipeline()


if __name__ == "__main__":
    main()
