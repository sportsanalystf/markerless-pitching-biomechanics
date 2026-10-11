"""PitchLab: pitching biomechanics from video, validated against marker-based capture.

The package has three parts that share one set of joint-centre definitions:

* ``pitchlab.data``       loads OpenBiomechanics (OBP) marker data and metrics
* ``pitchlab.geometry``,
  ``pitchlab.camera``,
  ``pitchlab.metrics``,
  ``pitchlab.events``     measure the same quantities from 3D markers or 2D pixels
* ``pitchlab.video``      runs a pose estimator on video and reports gated metrics

Because the video pipeline and the marker validation call the same metric code,
a number reported from video has been tested against marker truth under the
same definition, at the same camera view, and at the same frame rate.
"""

__version__ = "0.1.0"
