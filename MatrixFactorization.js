const Dui = [
    [1, 4, 5, 0, 3],
    [5, 1, 0, 5, 2],
    [0, 1, 0, 5, 0],
    [1, 0, 3, 0, 4],
    [2, 3, 0, 0, 4],
];
const k = 4;
const beta = 0.6;
const mU = 5;
const mI = 5;
let loop = 50000;

const w = [];
const h = [];

// fake W by random
for (let i = 0; i < mU; i++) {
    w.push([]);
    for (let j = 0; j < k; j++) {
        w[i][j] = Math.floor(Math.random() * 6);
    }
}
console.log('------------W--------------');
console.log(w);

// fake H by random
for (let i = 0; i < k; i++) {
    h.push([]);
    for (let j = 0; j < mI; j++) {
        h[i][j] = Math.floor(Math.random() * 6);
    }
}
console.log('------------H--------------');
console.log(h);

while (loop-- > 0) {
    // for ()
}
