export class ActionPredictor {
    constructor(convertsPathOrObject) {
        if (typeof convertsPathOrObject === "string") {
            // Load JSON from server (browser fetch)
            fetch(convertsPathOrObject)
                .then(res => res.json())
                .then(data => this.converts = data);
        } else {
            this.converts = convertsPathOrObject;
        }
    }

    async load() {
        const res = await fetch(this.path);
        this.converts = await res.json();
    }

    predict(frames){
        const counts = {};
        for (const f of frames) counts[f] = (counts[f] || 0) + 1;
        let best = null, max = 0;
        for (const [k, v] of Object.entries(counts)) {
            if (v > max) { max = v; best = k; }
        }
        if (max / frames.length >= 0.6) return best;
        return null;
    }

    removeDuplicates(arr) {
        const seen = new Set();
        return arr.filter(x => !seen.has(x) && seen.add(x));
    }

    actionSetToText(actionSet) {
        if (!this.converts) return null;

        let bestMatch = "";
        let bestLength = 0;

        // Helper to check if target is a subsequence of arr
        const isSubsequence = (arr, target) => {
            let i = 0;
            for (let t of target) {
                while (i < arr.length && arr[i] !== t) i++;
                if (i === arr.length) return false;
                i++;
            }
            return true;
        };

        for (const [text, actions] of Object.entries(this.converts)) {
            const filteredActions = actions.filter(a => a); // remove falsy values
            if (isSubsequence(actionSet, filteredActions)) {
                // pick the largest matching sequence
                if (filteredActions.length > bestLength) {
                    bestLength = filteredActions.length;
                    bestMatch = text;
                }
            }
        }

        return bestMatch || null; // return null if no match
    }


}
