import os
import sys

from rest_framework import viewsets, permissions, status
from rest_framework.response import Response
from rest_framework.decorators import action

# Add project root to path
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
sys.path.append(BASE_DIR)

from src.nlp.inference import ModelService
from .models import Prediction
from .serializers import PredictionSerializer

model_service = ModelService()


class PredictionViewSet(viewsets.ModelViewSet):

    serializer_class = PredictionSerializer
    permission_classes = [permissions.IsAuthenticated]

    def get_queryset(self):
        # Each user sees only their own predictions
        return Prediction.objects.filter(
            user=self.request.user
        ).order_by("-created_at")

    def perform_create(self, serializer):
        serializer.save(user=self.request.user)

    @action(detail=False, methods=["post"])
    def predict(self, request):

        text = request.data.get("text")

        if not text:
            return Response(
                {"error": "Text is required"},
                status=status.HTTP_400_BAD_REQUEST
            )

        predictions = model_service.predict(text)

        prediction_obj = Prediction.objects.create(
            user=request.user,
            input_text=text,
            result_json=predictions
        )

        serializer = self.get_serializer(prediction_obj)

        return Response({
            "status": "success",
            "prediction": serializer.data
        })