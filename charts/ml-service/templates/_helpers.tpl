{{- define "ml-service.name" -}}ml-service{{- end -}}
{{- define "ml-service.fullname" -}}{{ .Release.Name }}-{{ include "ml-service.name" . }}{{- end -}}
{{- define "ml-service.labels" -}}
app: {{ include "ml-service.name" . }}
app.kubernetes.io/name: {{ include "ml-service.name" . }}
app.kubernetes.io/instance: {{ .Release.Name }}
app.kubernetes.io/version: {{ .Chart.AppVersion | quote }}
app.kubernetes.io/managed-by: {{ .Release.Service }}
{{- end -}}
{{- define "ml-service.selector" -}}
app: {{ include "ml-service.name" . }}
app.kubernetes.io/instance: {{ .Release.Name }}
{{- end -}}
{{- define "ml-service.image" -}}{{ .Values.image.repository }}:{{ .Values.image.tag }}{{- end -}}
