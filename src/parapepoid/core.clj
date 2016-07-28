(ns parapepoid.core
  (:require [clojure.core.matrix :as matrix]
            [parapepoid.color :as c]
            [quil.core :as q]
            [quil.middleware :as mid]
            [thi.ng.geom.vector :as v]
            [thi.ng.math.core :as m]))

(matrix/set-current-implementation :vectorz)

(def config {:number-of-colors 3
             :number-of-samples 100
             :shape-radius 540
             :infinite-params {:hue 0.08
                               :saturation 0.2
                               :brightness 0.0}})

(defn regenerate [context]
  (let [palette (c/random-pallete (:number-of-colors config))
        y (/ (q/height) 2)
        xmax (q/width)
        radius (:shape-radius config)
        shapefn
        (fn [color]
          (let [center (v/vec2 (int (rand xmax)) y)]
            {:color color
             :center center
             :radius (int (rand radius))}))
        params (:infinite-params config)
        shapes (map shapefn (take (:number-of-samples config)
                                  (c/infinite-palette palette params)))
        sorted-shapes (reverse (sort-by :radius shapes))]
    (assoc context :shapes sorted-shapes)))

(defn- shape-to-rect [center radius]
  (let [x (- (:x center) (/ radius 2))
        y (- (:y center) (/ radius 2))]
    [x y radius radius]))

(defn setup []
  (q/frame-rate 10)
  (regenerate {}))

(defn update-context [context]
  context)

(defn draw [context]
  (q/background 240)
  (q/no-stroke)
  (doseq [shape (:shapes context)]
    (apply q/fill (c/as-rgb255-vec (:color shape)))
    (apply q/rect (shape-to-rect (:center shape) (:radius shape)))))

(defn key-pressed [context key-info]
  (case (:key key-info)
    :r (regenerate context)
    context))

(q/defsketch parapepoid
             :title "PARAPEPOID IS WATCHING YOU"
             :settings #(q/smooth 2)
             :setup setup
             :update update-context
             :draw draw
             :key-pressed key-pressed
             :size [1200 600]
             :middleware [mid/fun-mode])