# One input schema for CSV and packet observations

The schema-2 XGBoost upgrade retains the original **170 features and 84 category
names**. The application selects it only when its versioned bundle explicitly
records promotion over the unchanged checkpoint on the seven-count validation
profile. The original full-input checkpoint and binary packet model are retained.
See `PARTIAL_TRAINING.md` for the CUDA study and its separately recorded scores.

## Running the source application

From the project environment, run `python run_app.py` for the local development
interface or `python run_desktop.py` for Electron. The packaged Windows installer
embeds its own CPU inference runtime and small offline language model. It does
not need the training dataset, a Python installation, CUDA or an API key.

Open **Live screening**, choose the actual network adapter and **Packet
inspection**, then press **Start monitoring**. Windows requires Npcap. Completed
ten-second observations supply a shared vector directly to multiclass inference;
there is no intermediate CSV disk write. **Download feature CSV** exports exactly
that observation with all expected column names. Upload it in **CSV analysis**
to reproduce its vector and prediction using the same checkpoint.

Connection-counter mode continues to use separate observable-behavior rules.
It does not execute the multiclass packet classifier. Empty packet windows have
no model prediction. Delayed windows outside the existing one-second tolerance
are skipped. Each completed observation has a unique ID and timestamp; stale
export/XAI requests fail instead of silently using a newer window. Stop, capture
errors and stale observations clear current model verdicts.

## Canonical live measurements

| Header measurement | Existing CSV column | Unit |
| --- | --- | --- |
| All captured packets involving the selected local adapter | `network_packets_all_count` | Count per completed window |
| TCP ACK bit set | `network_tcp-flags-ack_count` | Count per completed window |
| TCP FIN bit set | `network_tcp-flags-fin_count` | Count per completed window |
| TCP PSH bit set | `network_tcp-flags-psh_count` | Count per completed window |
| TCP RST bit set | `network_tcp-flags-rst_count` | Count per completed window |
| TCP SYN bit set, including SYN+ACK | `network_tcp-flags-syn_count` | Count per completed window |
| TCP URG bit set | `network_tcp-flags-urg_count` | Count per completed window |

Every flag is counted independently. The other **163 inputs remain NaN**. No
sensor/log values, application protocols, packet-size aggregates or bidirectional
flow features are inferred from insufficient evidence. This implementation
follows the published count descriptions; equivalence with the publisher's
unavailable exact extraction code is not established.

## Partial CSV rules

- Columns are aligned by their exact frozen names, not their upload order.
- A missing column or blank value stays missing. A measured zero stays zero.
- Prepared protocol flags accept `0`, `1` or blank; missing does not mean absent.
- A raw protocol field such as `network_protocols_all` accepts a bounded list
  string. Missing/blank means unobserved; `[]` means an observed empty list and
  becomes zero flags. Observed unknown tokens use the frozen `UNSEEN` flag.
- Supply either a raw protocol list or its prepared flags for each protocol
  group. Mixed representations of the same group are rejected.
- Duplicate/empty column names, malformed row widths, infinite/out-of-range
  numbers, invalid flags and rows with no usable recognized measurements fail
  validation. Labels and ordinary metadata are ignored; unknown `network_` or
  `log_` columns are rejected as likely schema mistakes.
- Uploads retain the existing limit of 500 rows and 10 MB.

The feature table shows all expected names, values, measured/missing states and
which seven have a live mapping. Arbitrary partial combinations are labelled
**unvalidated** even though missing-pattern augmentation was used during fitting.
The exact seven-count profile has source-dataset validation only; this is not
labelled live-PC accuracy.

## Explaining partial predictions

The top three categories are tentative closed-set hypotheses, with uncalibrated
model scores and an explicit checkpoint version. They neither identify an
unknown attack nor confirm a real incident. The separate binary packet model is
shown only as a labelled comparison, with its own different validation task.

Exact native TreeSHAP retains every contribution and checks reconstruction of
the predicted class's raw margin. Contributions for unavailable features are
marked **missing-value default routing**; they are not measured traffic evidence.
LIME uses only observed columns to construct its local surrogate and restores
every unavailable numeric and categorical input to NaN for each model call.
Its perturbations are synthetic explanations, not captured packets or training
observations. SHAP and LIME are not causal explanations.

The existing offline Qwen/RAG section receives the category hypotheses, coverage
limitations, measured SHAP evidence and a missing-routing contribution when
applicable. Actual language generation runs only on request, on two CPU threads,
with bounded retrieval and at most two grounding attempts. Failed generation
returns an error without a template impersonating an LLM. The deterministic
research template remains separately labelled.

## Evidence and limitations

The original split, labels and artifacts are preserved. Validation covers only
60 of the 84 classes in the exact-ten-second subset, and 764 validation vectors
also occur in training after projection to seven counts. These limitations
preclude treating a high source-validation result as cross-device or Windows
live accuracy. Test scoring remains locked.

The recorded input-ambiguity diagnostic finds 18 identical seven-count vectors
with conflicting validation labels, affecting 381 rows. A deterministic
classifier using only these seven counts must make at least 45 errors among
the 1,571 validation rows: the empirical label-consistency ceiling is 97.1356%.
This is an information-limit diagnostic for this fixed partition, not a model
score or a held-out accuracy estimate. It rules out 99.97% on this exact task
without richer measurements or changing its labels. No labels are changed.

`experiment/verify_partial_capture.py` records replay/capture checks separately
from classifier validation. Actual unlabelled PC traffic demonstrates execution
only. The original benign-source replay that triggered a binary false alarm is
preserved. No packet payload archive, external AI service, automatic network
mitigation or synthetic replacement for a failed capture is added.
