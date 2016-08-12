(ns parapepoid.sketches.color-generation
  (:require [clojure.core.matrix :as matrix]
            [parapepoid.color.core :as c]
            [parapepoid.color.learn :as l]
            [parapepoid.nn.propagation :as p]
            [parapepoid.serialization :as s]
            [quil.core :as q]
            [quil.middleware :as mid]
            [thi.ng.geom.vector :as v]))

(matrix/set-current-implementation :vectorz)

(def config {; Display params
             :number-of-colors 3
             :number-of-samples 100
             :shape-radius 540
             :infinite-params {:hue 0.08
                               :saturation 0.2
                               :brightness 0.0}

             ; Network generation params
             :training-file "TR-I3-O1-RAND.clj"
             :network-params {:hidden-sizes [6]
                              :learning-rate 0.02
                              :batch-size 100
                              :epochs 2
                              :error-fn :cross-entropy}
             :approval-max-iterations 100
             :approval-threshold 0.7})

(defn color-to-shape [color]
  (let [y (/ (q/height) 2)
        xmax (q/width)
        radius (:shape-radius config)
        center (v/vec2 (int (rand xmax)) y)]
    {:color color
     :center center
     :radius (int (rand radius))}))

(defn generate-network [context]
  (let [file (:training-file config)
        [training test] (l/prepare-data file 0.2)
        params (:network-params config)
        network (first (l/evaluate-hyper-params training test params))]
    (assoc context :network network)))

(defn palette-to-shapes [palette]
  (let [params (:infinite-params config)
        shapes (map color-to-shape (take (:number-of-samples config)
                                         (c/infinite-palette palette params)))]
    (reverse (sort-by :radius shapes))))

(defn palette-into-context [context palette]
  (assoc context
    :current-palette palette
    :shapes (palette-to-shapes palette)))

(defn approved-palette [context]
  (let [iterations (:approval-max-iterations config)
        num-colors (:number-of-colors config)
        network (:network context)]
    (loop [i 0
           palette (c/random-pallete num-colors)]
      (if (= i iterations)
        (do
          (println "NO COLORS APPROVED, random palette is shown.")
          (palette-into-context context palette))
        (let [result (->> (into [] l/flatten-hsl palette)
                          (p/propagate-forward network)
                          first)]
          (println "Propagation result: " result)
          (if (> result
                 (:approval-threshold config))
            (do
              (println "Accepting palette after " i " iterations.")
              (palette-into-context context palette))
            (recur (inc i) (c/random-pallete num-colors))))))))

(defn random-palette [context]
  (let [palette (c/random-pallete (:number-of-colors config))]
    (palette-into-context context palette)))

(defn- shape-to-rect [center radius]
  (let [x (- (:x center) (/ radius 2))
        y (- (:y center) (/ radius 2))]
    [x y radius radius]))

(defn save-training [context output]
  (let [palette (:current-palette context)
        training (conj (:training context) [palette output])]
    (s/write-training (:training-file config) training)
    (assoc context :training training)))

(defn setup []
  (q/frame-rate 10)
  (conj (-> {}
            random-palette
            generate-network)
        {:training (or (s/read-training (:training-file config)) [])}))

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
    :p (do
         (println "Saving palette into training data as NOT APPROVED.")
         (random-palette (save-training context 0)))
    :o (do
         (println "Palette is APPROVED, saving to training file.")
         (random-palette (save-training context 1)))
    :n (do
         (println "Generating new network.")
         (generate-network context))
    :a (do
         (println "Generating a new approved palette.")
         (approved-palette context))
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
