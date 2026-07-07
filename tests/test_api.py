"""API integration tests — TestClient with mocked classifier, no network."""

from __future__ import annotations

import pytest


# ---------------------------------------------------------------------------
# Health endpoint
# ---------------------------------------------------------------------------


class TestHealthEndpoint:
    def test_returns_200(self, test_app):
        response = test_app.get("/api/clip/health")
        assert response.status_code == 200

    def test_returns_json(self, test_app):
        response = test_app.get("/api/clip/health")
        body = response.json()
        assert body["status"] == "ok"
        assert body["model_loaded"] is True


# ---------------------------------------------------------------------------
# POST /api/clip/classify
# ---------------------------------------------------------------------------


class TestClassifyEndpoint:
    @pytest.fixture
    def valid_payload(self, sample_image_rgb):
        return {
            "document_id": "doc_test_001",
            "image_paths": [str(sample_image_rgb)],
        }

    def test_valid_request_returns_200(self, test_app, valid_payload):
        response = test_app.post("/api/clip/classify", json=valid_payload)
        assert response.status_code == 200

    def test_response_schema(self, test_app, valid_payload):
        response = test_app.post("/api/clip/classify", json=valid_payload)
        body = response.json()
        assert body["document_id"] == "doc_test_001"
        assert len(body["classifications"]) == 1
        assert "model_name" in body
        assert "model_device" in body

        c = body["classifications"][0]
        assert "image_path" in c
        assert "image_type" in c
        assert "confidence" in c
        assert 0.0 <= c["confidence"] <= 1.0

    def test_empty_image_paths_returns_422(self, test_app):
        payload = {"document_id": "doc_001", "image_paths": []}
        response = test_app.post("/api/clip/classify", json=payload)
        assert response.status_code == 422

    def test_missing_document_id_returns_422(self, test_app):
        payload = {"image_paths": ["/some/path.jpg"]}
        response = test_app.post("/api/clip/classify", json=payload)
        assert response.status_code == 422

    def test_too_many_images_returns_422(self, test_app, sample_image_rgb):
        payload = {
            "document_id": "doc_001",
            "image_paths": [str(sample_image_rgb)] * 101,
        }
        response = test_app.post("/api/clip/classify", json=payload)
        assert response.status_code == 422

    def test_debug_flag_populates_all_scores(self, test_app, valid_payload):
        response = test_app.post(
            "/api/clip/classify?debug=true",
            json=valid_payload,
        )
        body = response.json()
        c = body["classifications"][0]
        assert c["all_scores"] is not None
        assert isinstance(c["all_scores"], dict)
        assert len(c["all_scores"]) > 0

    def test_no_debug_flag_omits_all_scores(self, test_app, valid_payload):
        response = test_app.post("/api/clip/classify", json=valid_payload)
        body = response.json()
        c = body["classifications"][0]
        assert c["all_scores"] is None

    def test_multiple_images(self, test_app, sample_image_rgb, sample_image_gray):
        payload = {
            "document_id": "doc_001",
            "image_paths": [str(sample_image_rgb), str(sample_image_gray)],
        }
        response = test_app.post("/api/clip/classify", json=payload)
        assert response.status_code == 200
        body = response.json()
        assert len(body["classifications"]) == 2

    def test_nonexistent_image_in_batch(self, test_app, sample_image_rgb, tmp_path):
        nonexistent = tmp_path / "ghost.jpg"
        payload = {
            "document_id": "doc_001",
            "image_paths": [str(sample_image_rgb), str(nonexistent)],
        }
        response = test_app.post("/api/clip/classify", json=payload)
        assert response.status_code == 200  # partial success
        body = response.json()
        assert body["classifications"][1]["image_type"] == "error"

    def test_document_id_invalid(self, test_app):
        """Empty document_id should fail validation."""
        payload = {"document_id": "", "image_paths": ["/test.jpg"]}
        response = test_app.post("/api/clip/classify", json=payload)
        assert response.status_code == 422

    def test_cors_headers(self, test_app, valid_payload):
        response = test_app.options(
            "/api/clip/classify",
            headers={
                "Origin": "http://localhost:5173",
                "Access-Control-Request-Method": "POST",
            },
        )
        # OPTIONS preflight should succeed
        assert response.status_code == 200
