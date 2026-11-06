//! TEST ONLY. A frozen copy of the base verifier (artifacts/explainer_20261103/base, 3 Nov 2026 08:58 IST, test modules stripped) so that
//! the base-vs-new safety tests compare against the base verdict live instead of a recorded string (redesign round 3 review, finding 7).
//! Never used by production code; never edit it except to re-copy the snapshot.
#![allow(dead_code)]
pub mod packet; pub mod verify; pub mod evidence;
