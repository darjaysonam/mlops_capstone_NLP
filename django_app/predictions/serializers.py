from rest_framework import serializers

from .models import Prediction


class PredictionSerializer(serializers.ModelSerializer):

    class Meta:
        model = Prediction
        fields = ["id", "user", "input_text", "result_json", "created_at"]
        read_only_fields = ["user", "result_json", "created_at"]
