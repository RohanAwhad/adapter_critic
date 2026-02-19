from adapter_critic.config import AppConfig


def test_config_model_smoke() -> None:
    config = AppConfig.model_validate(
        {
            "served_models": {
                "served-direct": {
                    "mode": "direct",
                    "api": {"model": "api-model", "base_url": "https://api.example"},
                }
            }
        }
    )
    assert "served-direct" in config.served_models
