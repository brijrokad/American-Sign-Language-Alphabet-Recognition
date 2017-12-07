// TEST DATA
// Keyed by mocha test ID
// Python code for generating test data can be found in the matching jupyter notebook in folder `notebooks/`.

;(function() {
  var DATA = {
    'convolutional.ZeroPadding2D.0': {
      input: {
        shape: [3, 5, 2],
        data: [
          -0.570441,
          -0.454673,
          -0.285321,
          0.237249,
          0.282682,
          0.428035,
          0.160547,
          -0.332203,
          0.546391,
          0.272735,
          0.010827,
          -0.763164,
          -0.442696,
          0.381948,
          -0.676994,
          0.753553,
          -0.031788,
          0.915329,
          -0.738844,
          0.269075,
          0.434091,
          0.991585,
          -0.944288,
          0.258834,
          0.162138,
          0.565201,
          -0.492094,
          0.170854,
          -0.139788,
          -0.710674
        ]
      },
      expected: {
        shape: [5, 7, 2],
        data: [
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          -0.570441,
          -0.454673,
          -0.285321,
          0.237249,
          0.282682,
          0.428035,
          0.160547,
          -0.332203,
          0.546391,
          0.272735,
          0.0,
          0.0,
          0.0,
          0.0,
          0.010827,
          -0.763164,
          -0.442696,
          0.381948,
          -0.676994,
          0.753553,
          -0.031788,
          0.915329,
          -0.738844,
          0.269075,
          0.0,
          0.0,
          0.0,
          0.0,
          0.434091,
          0.991585,
          -0.944288,
          0.258834,
          0.162138,
          0.565201,
          -0.492094,
          0.170854,
          -0.139788,
          -0.710674,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0
        ]
      }
    },
    'convolutional.ZeroPadding2D.1': {
      input: {
        shape: [3, 5, 2],
        data: [
          0.275222,
          -0.793967,
          -0.468107,
          -0.841484,
          -0.295362,
          0.78175,
          0.068787,
          -0.261747,
          -0.625733,
          -0.042907,
          0.861141,
          0.85267,
          0.956439,
          0.717838,
          -0.99869,
          -0.963008,
          0.013277,
          -0.180306,
          0.832137,
          -0.385252,
          -0.524308,
          0.659706,
          -0.905127,
          0.526292,
          0.832569,
          0.084455,
          0.23838,
          -0.046178,
          -0.735871,
          0.776883
        ]
      },
      expected: {
        shape: [3, 7, 4],
        data: [
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.275222,
          -0.793967,
          0.0,
          0.0,
          -0.468107,
          -0.841484,
          0.0,
          0.0,
          -0.295362,
          0.78175,
          0.0,
          0.0,
          0.068787,
          -0.261747,
          0.0,
          0.0,
          -0.625733,
          -0.042907,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.861141,
          0.85267,
          0.0,
          0.0,
          0.956439,
          0.717838,
          0.0,
          0.0,
          -0.99869,
          -0.963008,
          0.0,
          0.0,
          0.013277,
          -0.180306,
          0.0,
          0.0,
          0.832137,
          -0.385252,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          -0.524308,
          0.659706,
          0.0,
          0.0,
          -0.905127,
          0.526292,
          0.0,
          0.0,
          0.832569,
          0.084455,
          0.0,
          0.0,
          0.23838,
          -0.046178,
          0.0,
          0.0,
          -0.735871,
          0.776883,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0
        ]
      }
    },
    'convolutional.ZeroPadding2D.2': {
      input: {
        shape: [2, 6, 4],
        data: [
          -0.989173,
          -0.133618,
          -0.505338,
          0.023259,
          0.503982,
          -0.303769,
          -0.436321,
          0.793911,
          0.416102,
          0.806405,
          -0.098342,
          -0.738022,
          -0.982676,
          0.805073,
          0.741244,
          -0.941634,
          -0.253526,
          -0.136544,
          -0.295772,
          0.207565,
          -0.517246,
          -0.686963,
          -0.176235,
          -0.354111,
          -0.862411,
          -0.969822,
          0.200074,
          0.290718,
          -0.038623,
          0.294839,
          0.247968,
          0.557946,
          -0.455596,
          0.6624,
          0.879529,
          -0.466772,
          0.40423,
          0.213794,
          0.645662,
          -0.044634,
          -0.552595,
          0.771242,
          -0.131944,
          -0.172725,
          0.700856,
          -0.001994,
          0.606737,
          -0.593306
        ]
      },
      expected: {
        shape: [8, 10, 4],
        data: [
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          -0.989173,
          -0.133618,
          -0.505338,
          0.023259,
          0.503982,
          -0.303769,
          -0.436321,
          0.793911,
          0.416102,
          0.806405,
          -0.098342,
          -0.738022,
          -0.982676,
          0.805073,
          0.741244,
          -0.941634,
          -0.253526,
          -0.136544,
          -0.295772,
          0.207565,
          -0.517246,
          -0.686963,
          -0.176235,
          -0.354111,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          -0.862411,
          -0.969822,
          0.200074,
          0.290718,
          -0.038623,
          0.294839,
          0.247968,
          0.557946,
          -0.455596,
          0.6624,
          0.879529,
          -0.466772,
          0.40423,
          0.213794,
          0.645662,
          -0.044634,
          -0.552595,
          0.771242,
          -0.131944,
          -0.172725,
          0.700856,
          -0.001994,
          0.606737,
          -0.593306,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0
        ]
      }
    },
    'convolutional.ZeroPadding2D.3': {
      input: {
        shape: [2, 6, 4],
        data: [
          -0.47588,
          0.366985,
          0.040173,
          0.015578,
          -0.906159,
          0.241982,
          -0.771299,
          -0.443554,
          -0.56404,
          -0.17751,
          0.541277,
          -0.233327,
          0.024369,
          0.858275,
          0.496191,
          0.980574,
          -0.59522,
          0.480899,
          0.392553,
          -0.191718,
          0.055121,
          0.289836,
          -0.498339,
          0.800408,
          0.132679,
          -0.716649,
          0.840092,
          -0.088837,
          -0.538209,
          -0.580887,
          -0.370128,
          -0.924933,
          -0.161736,
          -0.205619,
          0.793729,
          -0.354472,
          0.687519,
          0.272041,
          -0.943352,
          -0.730959,
          -0.330419,
          -0.479307,
          0.520387,
          0.137906,
          0.897598,
          0.869815,
          0.978562,
          0.731387
        ]
      },
      expected: {
        shape: [2, 12, 8],
        data: [
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          -0.47588,
          0.366985,
          0.040173,
          0.015578,
          0.0,
          0.0,
          0.0,
          0.0,
          -0.906159,
          0.241982,
          -0.771299,
          -0.443554,
          0.0,
          0.0,
          0.0,
          0.0,
          -0.56404,
          -0.17751,
          0.541277,
          -0.233327,
          0.0,
          0.0,
          0.0,
          0.0,
          0.024369,
          0.858275,
          0.496191,
          0.980574,
          0.0,
          0.0,
          0.0,
          0.0,
          -0.59522,
          0.480899,
          0.392553,
          -0.191718,
          0.0,
          0.0,
          0.0,
          0.0,
          0.055121,
          0.289836,
          -0.498339,
          0.800408,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.132679,
          -0.716649,
          0.840092,
          -0.088837,
          0.0,
          0.0,
          0.0,
          0.0,
          -0.538209,
          -0.580887,
          -0.370128,
          -0.924933,
          0.0,
          0.0,
          0.0,
          0.0,
          -0.161736,
          -0.205619,
          0.793729,
          -0.354472,
          0.0,
          0.0,
          0.0,
          0.0,
          0.687519,
          0.272041,
          -0.943352,
          -0.730959,
          0.0,
          0.0,
          0.0,
          0.0,
          -0.330419,
          -0.479307,
          0.520387,
          0.137906,
          0.0,
          0.0,
          0.0,
          0.0,
          0.897598,
          0.869815,
          0.978562,
          0.731387,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0
        ]
      }
    },
    'convolutional.ZeroPadding2D.4': {
      input: {
        shape: [2, 6, 4],
        data: [
          0.024124,
          0.280236,
          -0.680013,
          -0.042458,
          -0.164273,
          0.358409,
          0.511014,
          -0.585272,
          -0.481578,
          0.692702,
          0.64189,
          -0.400252,
          -0.922248,
          -0.735105,
          -0.533918,
          0.071402,
          0.310474,
          0.369868,
          0.767931,
          -0.842066,
          -0.091189,
          0.835301,
          -0.480484,
          0.950819,
          -0.002131,
          0.086491,
          -0.480947,
          0.405572,
          -0.083803,
          -0.921447,
          -0.291545,
          0.674087,
          -0.560444,
          0.881432,
          0.076544,
          0.63549,
          -0.185686,
          -0.89067,
          0.709257,
          -0.256164,
          -0.873627,
          0.330906,
          -0.583426,
          -0.51286,
          0.751485,
          0.030077,
          -0.998662,
          0.175588
        ]
      },
      expected: {
        shape: [5, 13, 4],
        data: [
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.024124,
          0.280236,
          -0.680013,
          -0.042458,
          -0.164273,
          0.358409,
          0.511014,
          -0.585272,
          -0.481578,
          0.692702,
          0.64189,
          -0.400252,
          -0.922248,
          -0.735105,
          -0.533918,
          0.071402,
          0.310474,
          0.369868,
          0.767931,
          -0.842066,
          -0.091189,
          0.835301,
          -0.480484,
          0.950819,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          -0.002131,
          0.086491,
          -0.480947,
          0.405572,
          -0.083803,
          -0.921447,
          -0.291545,
          0.674087,
          -0.560444,
          0.881432,
          0.076544,
          0.63549,
          -0.185686,
          -0.89067,
          0.709257,
          -0.256164,
          -0.873627,
          0.330906,
          -0.583426,
          -0.51286,
          0.751485,
          0.030077,
          -0.998662,
          0.175588,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0
        ]
      }
    },
    'convolutional.ZeroPadding2D.5': {
      input: {
        shape: [2, 6, 4],
        data: [
          -0.072127,
          -0.553929,
          -0.355552,
          -0.936405,
          0.556627,
          -0.482815,
          -0.225337,
          -0.640315,
          0.023246,
          -0.638412,
          -0.797304,
          0.284959,
          -0.569771,
          -0.685286,
          0.002481,
          0.398436,
          0.11345,
          0.416629,
          -0.526713,
          0.962183,
          0.021732,
          0.922994,
          0.07991,
          -0.164385,
          0.461494,
          -0.982877,
          -0.142158,
          0.175741,
          -0.124041,
          -0.875609,
          -0.528708,
          -0.911127,
          0.782257,
          -0.509403,
          0.573973,
          -0.151309,
          -0.895619,
          -0.721042,
          0.483952,
          -0.745814,
          -0.588825,
          -0.154089,
          0.423904,
          -0.262707,
          -0.517175,
          -0.535505,
          -0.266104,
          -0.46314
        ]
      },
      expected: {
        shape: [6, 10, 4],
        data: [
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          -0.072127,
          -0.553929,
          -0.355552,
          -0.936405,
          0.556627,
          -0.482815,
          -0.225337,
          -0.640315,
          0.023246,
          -0.638412,
          -0.797304,
          0.284959,
          -0.569771,
          -0.685286,
          0.002481,
          0.398436,
          0.11345,
          0.416629,
          -0.526713,
          0.962183,
          0.021732,
          0.922994,
          0.07991,
          -0.164385,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.461494,
          -0.982877,
          -0.142158,
          0.175741,
          -0.124041,
          -0.875609,
          -0.528708,
          -0.911127,
          0.782257,
          -0.509403,
          0.573973,
          -0.151309,
          -0.895619,
          -0.721042,
          0.483952,
          -0.745814,
          -0.588825,
          -0.154089,
          0.423904,
          -0.262707,
          -0.517175,
          -0.535505,
          -0.266104,
          -0.46314,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0
        ]
      }
    }
  }

  window.TEST_DATA = Object.assign({}, window.TEST_DATA, DATA)
})()
